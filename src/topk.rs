//! Bounded top-k that keeps the `k` smallest-distance entries. For small k
//! (typically 10) a flat array with linear max-tracking beats a heap.

pub struct TopK {
    k: usize,
    dists: Vec<f32>,
    ids: Vec<u32>,
    max_val: f32,
    max_pos: usize,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            dists: Vec::with_capacity(k),
            ids: Vec::with_capacity(k),
            max_val: f32::INFINITY,
            max_pos: 0,
        }
    }

    /// True if `dist` could enter the current top-k (cheap gate before push).
    #[inline(always)]
    pub fn would_accept(&self, dist: f32) -> bool {
        self.dists.len() < self.k || dist < self.max_val
    }

    #[inline]
    fn recompute_max(&mut self) {
        let mut mv = f32::NEG_INFINITY;
        let mut mp = 0;
        for (i, &dd) in self.dists.iter().enumerate() {
            if dd > mv {
                mv = dd;
                mp = i;
            }
        }
        self.max_val = mv;
        self.max_pos = mp;
    }

    #[inline]
    pub fn push(&mut self, dist: f32, id: u32) {
        if self.dists.len() < self.k {
            self.dists.push(dist);
            self.ids.push(id);
            if self.dists.len() == self.k {
                self.recompute_max();
            }
        } else if dist < self.max_val {
            self.dists[self.max_pos] = dist;
            self.ids[self.max_pos] = id;
            self.recompute_max();
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (f32, u32)> + '_ {
        self.dists.iter().copied().zip(self.ids.iter().copied())
    }

    /// Consume into ascending-by-distance (dist, id) pairs.
    pub fn into_sorted(self) -> Vec<(f32, u32)> {
        let mut v: Vec<(f32, u32)> = self
            .dists
            .into_iter()
            .zip(self.ids.into_iter())
            .collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}
