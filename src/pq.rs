//! Product quantization. The vector (residual after subtracting the coarse
//! centroid) is split into `m` sub-vectors of dim `dsub`; each sub-vector is
//! encoded to the nearest of `ksub` (=256) sub-centroids, giving an `m`-byte
//! code. Search uses Asymmetric Distance Computation (ADC): a per-query lookup
//! table of size `m * ksub` reduces each code's distance to `m` table adds.

use crate::kmeans::kmeans;
use crate::math::{argmin_blk, to_blocks};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct PqParams {
    pub m: usize,
    pub ksub: usize,
    pub dsub: usize,
}

pub struct Pq {
    pub p: PqParams,
    pub codebook: Vec<f32>,   // m * ksub * dsub (row-major per subspace)
    codebook_b: Vec<f32>,     // m * (block16 layout) per subspace, for fast encode
}

fn build_codebook_b(codebook: &[f32], m: usize, ksub: usize, dsub: usize) -> Vec<f32> {
    let stride = ksub * dsub;
    let mut out = vec![0.0f32; m * stride];
    for s in 0..m {
        let src = &codebook[s * stride..(s + 1) * stride];
        let b = to_blocks(src, ksub, dsub); // ksub x dsub -> block16
        out[s * stride..(s + 1) * stride].copy_from_slice(&b);
    }
    out
}

impl Pq {
    pub fn d(&self) -> usize {
        self.p.m * self.p.dsub
    }

    /// Train `m` independent sub-quantizers on residual training data
    /// (`n * d` row-major, d = m*dsub).
    pub fn train(residuals: &[f32], n: usize, d: usize, m: usize, ksub: usize, iters: usize, seed: u64) -> Pq {
        assert_eq!(d % m, 0);
        let dsub = d / m;
        let mut codebook = vec![0.0f32; m * ksub * dsub];

        let subbooks: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|s| {
                // Gather subspace columns [s*dsub, (s+1)*dsub) contiguously.
                let mut sub = vec![0.0f32; n * dsub];
                for i in 0..n {
                    let src = &residuals[i * d + s * dsub..i * d + s * dsub + dsub];
                    sub[i * dsub..i * dsub + dsub].copy_from_slice(src);
                }
                kmeans(&sub, n, dsub, ksub, iters, seed ^ (s as u64 + 1))
            })
            .collect();

        for s in 0..m {
            codebook[s * ksub * dsub..(s + 1) * ksub * dsub].copy_from_slice(&subbooks[s]);
        }
        let codebook_b = build_codebook_b(&codebook, m, ksub, dsub);
        Pq {
            p: PqParams { m, ksub, dsub },
            codebook,
            codebook_b,
        }
    }

    pub fn from_parts(p: PqParams, codebook: Vec<f32>) -> Pq {
        let codebook_b = build_codebook_b(&codebook, p.m, p.ksub, p.dsub);
        Pq { p, codebook, codebook_b }
    }

    #[inline(always)]
    fn subbook(&self, s: usize) -> &[f32] {
        let stride = self.p.ksub * self.p.dsub;
        &self.codebook[s * stride..(s + 1) * stride]
    }

    #[inline(always)]
    fn subbook_b(&self, s: usize) -> &[f32] {
        let stride = self.p.ksub * self.p.dsub;
        &self.codebook_b[s * stride..(s + 1) * stride]
    }

    /// Encode one residual vector (length d) into an `m`-byte code.
    #[inline]
    pub fn encode_into(&self, residual: &[f32], code: &mut [u8]) {
        let dsub = self.p.dsub;
        let ksub = self.p.ksub;
        for s in 0..self.p.m {
            let sub = &residual[s * dsub..s * dsub + dsub];
            let (idx, _) = argmin_blk(sub, self.subbook_b(s), ksub, dsub);
            code[s] = idx as u8;
        }
    }

    /// Build the ADC lookup table for a query residual: `m * ksub` distances.
    pub fn build_lut(&self, residual: &[f32], lut: &mut [f32]) {
        let dsub = self.p.dsub;
        let ksub = self.p.ksub;
        for s in 0..self.p.m {
            let sub = &residual[s * dsub..s * dsub + dsub];
            let book = self.subbook(s);
            let out = &mut lut[s * ksub..(s + 1) * ksub];
            for c in 0..ksub {
                let cc = &book[c * dsub..c * dsub + dsub];
                let mut acc = 0.0f32;
                for j in 0..dsub {
                    let diff = sub[j] - cc[j];
                    acc += diff * diff;
                }
                out[c] = acc;
            }
        }
    }
}

/// Sum an ADC lookup over a packed code (`m` bytes). Hot inner loop of search.
#[inline(always)]
pub fn adc_distance(lut: &[f32], code: &[u8], m: usize, ksub: usize) -> f32 {
    let mut acc = 0.0f32;
    for s in 0..m {
        acc += lut[s * ksub + code[s] as usize];
    }
    acc
}
