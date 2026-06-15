//! Small, fast, fully-deterministic PRNGs.
//!
//! Determinism matters: the billion-vector dataset is generated on the fly from
//! a vector index `i` (we never store the raw float32 vectors, which would be
//! ~512 GB). Both the "assign" pass and the "encode" pass must regenerate the
//! exact same vector for a given `i`, so generation is a pure function of
//! `(seed, i)`.

#[inline(always)]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// xoshiro256++ — fast, high quality, used for per-vector noise streams.
#[derive(Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    #[inline]
    pub fn seed(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            *v = splitmix64(&mut sm);
        }
        Rng { s }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f32 in [0, 1).
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        // top 24 bits -> [0,1)
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
    }

    /// Standard-normal-ish sample via sum of 4 uniforms (Irwin–Hall, cheap and
    /// good enough for synthetic clustered data). Mean 0, unit-ish variance.
    #[inline(always)]
    pub fn next_gaussian(&mut self) -> f32 {
        let s = self.next_f32() + self.next_f32() + self.next_f32() + self.next_f32();
        // Irwin-Hall(4): mean 2, var 4/12=1/3 -> normalize to ~unit variance.
        (s - 2.0) * 1.732_050_8
    }

    #[inline(always)]
    pub fn next_below(&mut self, bound: u64) -> u64 {
        // Lemire-ish; bias negligible for our use.
        ((self.next_u64() as u128 * bound as u128) >> 64) as u64
    }
}
