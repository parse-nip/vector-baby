//! Plain Lloyd k-means with parallel assignment. Used both for the coarse
//! quantizer (IVF centroids) and for each product-quantization subspace.

use crate::math::argmin_l2;
use crate::rng::Rng;
use rayon::prelude::*;

/// Returns `k * d` centroids. `data` is `n * d` row-major.
pub fn kmeans(data: &[f32], n: usize, d: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    assert!(n >= k, "need at least k points to train k centroids");
    let mut centroids = vec![0.0f32; k * d];

    // Init: pick k distinct random points.
    let mut r = Rng::seed(seed);
    let mut chosen = vec![false; n];
    for c in 0..k {
        let mut idx = r.next_below(n as u64) as usize;
        let mut guard = 0;
        while chosen[idx] && guard < 64 {
            idx = r.next_below(n as u64) as usize;
            guard += 1;
        }
        chosen[idx] = true;
        centroids[c * d..c * d + d].copy_from_slice(&data[idx * d..idx * d + d]);
    }

    let nthreads = rayon::current_num_threads().max(1);
    for _ in 0..iters {
        // Parallel assignment + partial accumulation per thread.
        let partials: Vec<(Vec<f64>, Vec<u64>)> = (0..n)
            .into_par_iter()
            .fold(
                || (vec![0.0f64; k * d], vec![0u64; k]),
                |mut acc, i| {
                    let x = &data[i * d..i * d + d];
                    let (c, _) = argmin_l2(x, &centroids, d);
                    let base = c * d;
                    for j in 0..d {
                        acc.0[base + j] += x[j] as f64;
                    }
                    acc.1[c] += 1;
                    acc
                },
            )
            .collect();

        let mut sums = vec![0.0f64; k * d];
        let mut counts = vec![0u64; k];
        for (s, cnt) in &partials {
            for j in 0..k * d {
                sums[j] += s[j];
            }
            for c in 0..k {
                counts[c] += cnt[c];
            }
        }
        let _ = nthreads;

        for c in 0..k {
            if counts[c] == 0 {
                // Empty cluster: reseed from a random data point.
                let idx = r.next_below(n as u64) as usize;
                centroids[c * d..c * d + d].copy_from_slice(&data[idx * d..idx * d + d]);
            } else {
                let inv = 1.0 / counts[c] as f64;
                for j in 0..d {
                    centroids[c * d + j] = (sums[c * d + j] * inv) as f32;
                }
            }
        }
    }
    centroids
}
