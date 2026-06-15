//! Flat (exact) cosine index for small/medium collections.
//!
//! For the BAYC POC (10k vectors) exact search is the right call: it's a few
//! microseconds per query, gives perfect recall, and avoids any quantization
//! loss that would blur fine semantic distinctions. The IVF-PQ index in
//! `index.rs` is the path for billion-scale corpora; this is the path for a
//! single collection.
//!
//! CLIP embeddings are L2-normalized, so ranking by L2 distance is identical to
//! ranking by cosine similarity (`||a-b||^2 = 2 - 2*cos`). We reuse the same
//! AVX2 `l2_sq` kernel and report cosine = 1 - l2/2 for display.

use crate::math::l2_sq;
use crate::topk::TopK;
use rayon::prelude::*;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct NftMeta {
    d: usize,
    #[serde(default)]
    model: String,
    tokens: Vec<u32>,
}

pub struct FlatIndex {
    pub d: usize,
    pub model: String,
    pub tokens: Vec<u32>,
    pub vecs: Vec<f32>, // n*d, L2-normalized
}

impl FlatIndex {
    pub fn open(dir: &Path) -> std::io::Result<FlatIndex> {
        let meta: NftMeta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
        let bytes = std::fs::read(dir.join("embeddings.f32"))?;
        let nfloats = bytes.len() / 4;
        let mut vecs = vec![0.0f32; nfloats];
        let dst = unsafe { std::slice::from_raw_parts_mut(vecs.as_mut_ptr() as *mut u8, nfloats * 4) };
        dst.copy_from_slice(&bytes[..nfloats * 4]);
        assert_eq!(nfloats, meta.tokens.len() * meta.d, "embeddings/meta size mismatch");
        Ok(FlatIndex {
            d: meta.d,
            model: meta.model,
            tokens: meta.tokens,
            vecs,
        })
    }

    pub fn n(&self) -> usize {
        self.tokens.len()
    }

    /// Exact top-k by cosine. `query` must be L2-normalized and length `d`.
    /// Returns (token_id, cosine_similarity) descending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let d = self.d;
        let n = self.n();
        let top = (0..n)
            .into_par_iter()
            .fold(
                || TopK::new(k),
                |mut acc, i| {
                    let dist = l2_sq(query, &self.vecs[i * d..i * d + d]);
                    if acc.would_accept(dist) {
                        acc.push(dist, i as u32);
                    }
                    acc
                },
            )
            .reduce(
                || TopK::new(k),
                |mut a, b| {
                    for (dd, id) in b.iter() {
                        a.push(dd, id);
                    }
                    a
                },
            );
        top.into_sorted()
            .into_iter()
            .map(|(l2, i)| (self.tokens[i as usize], 1.0 - l2 / 2.0))
            .collect()
    }
}
