//! Deterministic synthetic dataset.
//!
//! We never materialize the full billion-vector matrix (that would be ~512 GB
//! at d=128). Instead every vector is a pure function of its index `i`:
//! it belongs to one of `num_centers` Gaussian cluster centers plus a small
//! per-dimension noise term. This gives data with genuine nearest-neighbor
//! structure (so recall is meaningful) while letting us regenerate any vector
//! on demand during the multi-pass build.

use crate::rng::{splitmix64, Rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct DatasetSpec {
    pub d: usize,
    pub num_centers: u64,
    pub center_scale: f32,
    pub noise: f32,
    pub seed: u64,
}

pub struct Dataset {
    pub spec: DatasetSpec,
    centers: Vec<f32>, // num_centers * d
}

#[inline(always)]
fn vec_seed(seed: u64, i: u64) -> u64 {
    let mut s = seed ^ i.wrapping_mul(0x9E3779B97F4A7C15);
    splitmix64(&mut s);
    splitmix64(&mut s)
}

impl Dataset {
    pub fn new(spec: DatasetSpec) -> Self {
        let d = spec.d;
        let nc = spec.num_centers as usize;
        let mut centers = vec![0.0f32; nc * d];
        let seed = spec.seed;
        let scale = spec.center_scale;
        centers
            .par_chunks_mut(d)
            .enumerate()
            .for_each(|(c, out)| {
                let mut r = Rng::seed(seed ^ 0xD1B54A32D192ED03 ^ (c as u64).wrapping_mul(0x9E3779B97F4A7C15));
                for x in out.iter_mut() {
                    *x = r.next_gaussian() * scale;
                }
            });
        Dataset { spec, centers }
    }

    #[inline(always)]
    pub fn d(&self) -> usize {
        self.spec.d
    }

    /// Generate vector `i` into `out` (length d). Returns the source center id.
    #[inline(always)]
    pub fn gen(&self, i: u64, out: &mut [f32]) -> u64 {
        let d = self.spec.d;
        let mut r = Rng::seed(vec_seed(self.spec.seed, i));
        let c = r.next_below(self.spec.num_centers);
        let base = (c as usize) * d;
        let noise = self.spec.noise;
        let center = &self.centers[base..base + d];
        for k in 0..d {
            out[k] = center[k] + (r.next_f32() * 2.0 - 1.0) * noise;
        }
        c
    }

    /// Generate a contiguous block [start, start+count) into `out`
    /// (row-major, count*d), in parallel.
    pub fn gen_block(&self, start: u64, count: usize, out: &mut [f32]) {
        let d = self.spec.d;
        debug_assert_eq!(out.len(), count * d);
        out.par_chunks_mut(d).enumerate().for_each(|(j, row)| {
            self.gen(start + j as u64, row);
        });
    }
}

/// A query set with planted ground-truth: each query is a known base vector
/// `target` plus tiny noise, so `target` is (almost surely) its true nearest
/// neighbor. Lets us measure recall at billion-scale without a 512 GB brute
/// force pass.
#[derive(Clone, Serialize, Deserialize)]
pub struct QuerySet {
    pub d: usize,
    pub vectors: Vec<f32>, // nq * d
    pub targets: Vec<u64>, // nq
}

impl QuerySet {
    pub fn nq(&self) -> usize {
        self.targets.len()
    }
    pub fn query(&self, j: usize) -> &[f32] {
        &self.vectors[j * self.d..(j + 1) * self.d]
    }
}

pub fn make_queries(ds: &Dataset, n: u64, nq: usize, query_noise: f32, qseed: u64) -> QuerySet {
    let d = ds.d();
    let mut vectors = vec![0.0f32; nq * d];
    let mut targets = vec![0u64; nq];
    let mut tmp = vec![0.0f32; d];
    for j in 0..nq {
        let mut r = Rng::seed(qseed ^ (j as u64).wrapping_mul(0x100000001B3));
        let t = r.next_below(n);
        targets[j] = t;
        ds.gen(t, &mut tmp);
        let row = &mut vectors[j * d..(j + 1) * d];
        for k in 0..d {
            row[k] = tmp[k] + (r.next_f32() * 2.0 - 1.0) * query_noise;
        }
    }
    QuerySet { d, vectors, targets }
}
