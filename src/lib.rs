pub mod dataset;
pub mod index;
pub mod kmeans;
pub mod math;
pub mod pq;
pub mod rng;
pub mod server;
pub mod topk;

use dataset::{Dataset, QuerySet};
use math::l2_sq;
use rayon::prelude::*;
use topk::TopK;

/// Query-set generation constants so `bench`/`serve`/`recall` all agree.
pub const QUERY_NOISE: f32 = 0.02;
pub fn qseed(seed: u64) -> u64 {
    seed ^ 0x9999_1234_ABCD_0001
}

/// Exact top-k for every query in a single streaming pass over the dataset.
/// Used to measure true recall at scales where brute force is feasible.
pub fn brute_force_all(
    ds: &Dataset,
    n: u64,
    queries: &QuerySet,
    k: usize,
    batch: usize,
) -> Vec<Vec<(f32, u32)>> {
    let d = ds.d();
    let nq = queries.nq();
    let mut tops: Vec<TopK> = (0..nq).map(|_| TopK::new(k)).collect();
    let mut vecs = vec![0.0f32; batch * d];
    let mut done: u64 = 0;
    while done < n {
        let cur = ((n - done) as usize).min(batch);
        ds.gen_block(done, cur, &mut vecs[..cur * d]);
        tops.par_iter_mut().enumerate().for_each(|(qi, top)| {
            let q = queries.query(qi);
            for p in 0..cur {
                let x = &vecs[p * d..p * d + d];
                let dist = l2_sq(q, x);
                if top.would_accept(dist) {
                    top.push(dist, (done + p as u64) as u32);
                }
            }
        });
        done += cur as u64;
    }
    tops.into_iter().map(|t| t.into_sorted()).collect()
}
