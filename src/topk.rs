//! Bounded top-k keeping the `k` smallest-distance entries, backed by a binary
//! max-heap so `push` is O(log k). This matters because the IVF-PQ candidate
//! shortlist can be large (thousands) at billion scale; an O(k) scheme would
//! dominate the scan.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Item(f32, u32);

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap on distance: the largest distance is at the top so it can be
        // evicted when a closer candidate arrives.
        self.0.total_cmp(&other.0)
    }
}

pub struct TopK {
    k: usize,
    heap: BinaryHeap<Item>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    #[inline(always)]
    pub fn would_accept(&self, dist: f32) -> bool {
        self.heap.len() < self.k || dist < self.heap.peek().map(|i| i.0).unwrap_or(f32::INFINITY)
    }

    #[inline]
    pub fn push(&mut self, dist: f32, id: u32) {
        if self.heap.len() < self.k {
            self.heap.push(Item(dist, id));
        } else if let Some(top) = self.heap.peek() {
            if dist < top.0 {
                self.heap.pop();
                self.heap.push(Item(dist, id));
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (f32, u32)> + '_ {
        self.heap.iter().map(|i| (i.0, i.1))
    }

    /// Consume into ascending-by-distance (dist, id) pairs.
    pub fn into_sorted(self) -> Vec<(f32, u32)> {
        let mut v: Vec<(f32, u32)> = self.heap.into_iter().map(|i| (i.0, i.1)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }
}
