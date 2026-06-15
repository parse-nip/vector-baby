//! Distance kernels.
//!
//! We target **AVX2 + FMA**, which are stable on this Rust toolchain (AVX-512
//! intrinsics are still unstable on stable Rust, so we don't rely on them).
//! AVX2 processes 8 f32 lanes per instruction; with FMA and multiple
//! independent accumulators this gets the build kernels close to core peak.

/// SIMD lane count used by the block-interleaved centroid layout.
pub const LANES: usize = 8;

#[cfg(target_feature = "avx2")]
#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    unsafe { l2_sq_avx2(a, b) }
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        let d1 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)));
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        i += 16;
    }
    while i + 8 <= n {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        i += 8;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    while i < n {
        let d = *pa.add(i) - *pb.add(i);
        s += d * d;
        i += 1;
    }
    s
}

#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        let base = c * 8;
        for j in 0..8 {
            let diff = a[base + j] - b[base + j];
            acc[j] += diff * diff;
        }
    }
    let mut s = 0.0f32;
    for j in 0..8 {
        s += acc[j];
    }
    for k in (chunks * 8)..a.len() {
        let diff = a[k] - b[k];
        s += diff * diff;
    }
    s
}

/// Nearest of `k` centroids (row-major, k*d) to `x`. Returns (index, dist_sq).
#[inline]
pub fn argmin_l2(x: &[f32], centroids: &[f32], d: usize) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    let k = centroids.len() / d;
    for c in 0..k {
        let dist = l2_sq(x, &centroids[c * d..c * d + d]);
        if dist < best_d {
            best_d = dist;
            best = c;
        }
    }
    (best, best_d)
}

/// Re-pack row-major centroids (`k x d`, k a multiple of [`LANES`]) into a
/// block-interleaved layout: `LANES` centroids per block, values stored
/// `[dim][lane]`, i.e. `cb[b*d*LANES + dim*LANES + lane]` holds dimension
/// `dim` of centroid `b*LANES + lane`.
///
/// This gives the across-centroids distance kernel fully contiguous loads
/// (one vector load per dim) while vectorizing `LANES` centroids at a time —
/// good ILP *and* good locality, unlike a plain transpose (whose per-dim page
/// stride wrecks the TLB).
pub fn to_blocks(src: &[f32], k: usize, d: usize) -> Vec<f32> {
    assert!(k % LANES == 0, "k must be a multiple of {}", LANES);
    let mut out = vec![0.0f32; k * d];
    for c in 0..k {
        let b = c / LANES;
        let lane = c % LANES;
        for dim in 0..d {
            out[b * d * LANES + dim * LANES + lane] = src[c * d + dim];
        }
    }
    out
}

/// Nearest centroid over the block-interleaved layout from [`to_blocks`].
/// Returns (centroid_index, dist_sq).
#[cfg(target_feature = "avx2")]
#[inline]
pub fn argmin_blk(query: &[f32], cb: &[f32], k: usize, d: usize) -> (u32, f32) {
    unsafe { argmin_blk_avx2(query, cb, k, d) }
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2,fma")]
unsafe fn argmin_blk_avx2(query: &[f32], cb: &[f32], k: usize, d: usize) -> (u32, f32) {
    use std::arch::x86_64::*;
    let mut best: u32 = 0;
    let mut bestd = f32::INFINITY;
    let qp = query.as_ptr();
    let mut tmp = [0.0f32; 8];
    let nblk = k / 8;

    for b in 0..nblk {
        let base = cb.as_ptr().add(b * d * 8);
        // 4 independent accumulator chains hide FMA latency.
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut dim = 0usize;
        while dim + 4 <= d {
            let q0 = _mm256_set1_ps(*qp.add(dim));
            let e0 = _mm256_sub_ps(q0, _mm256_loadu_ps(base.add(dim * 8)));
            a0 = _mm256_fmadd_ps(e0, e0, a0);
            let q1 = _mm256_set1_ps(*qp.add(dim + 1));
            let e1 = _mm256_sub_ps(q1, _mm256_loadu_ps(base.add((dim + 1) * 8)));
            a1 = _mm256_fmadd_ps(e1, e1, a1);
            let q2 = _mm256_set1_ps(*qp.add(dim + 2));
            let e2 = _mm256_sub_ps(q2, _mm256_loadu_ps(base.add((dim + 2) * 8)));
            a2 = _mm256_fmadd_ps(e2, e2, a2);
            let q3 = _mm256_set1_ps(*qp.add(dim + 3));
            let e3 = _mm256_sub_ps(q3, _mm256_loadu_ps(base.add((dim + 3) * 8)));
            a3 = _mm256_fmadd_ps(e3, e3, a3);
            dim += 4;
        }
        while dim < d {
            let q0 = _mm256_set1_ps(*qp.add(dim));
            let e0 = _mm256_sub_ps(q0, _mm256_loadu_ps(base.add(dim * 8)));
            a0 = _mm256_fmadd_ps(e0, e0, a0);
            dim += 1;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let off = b * 8;
        for l in 0..8 {
            if tmp[l] < bestd {
                bestd = tmp[l];
                best = (off + l) as u32;
            }
        }
    }
    (best, bestd)
}

#[cfg(not(target_feature = "avx2"))]
#[inline]
pub fn argmin_blk(query: &[f32], cb: &[f32], k: usize, d: usize) -> (u32, f32) {
    let mut best = 0u32;
    let mut bestd = f32::INFINITY;
    let nblk = k / LANES;
    for b in 0..nblk {
        let base = b * d * LANES;
        for lane in 0..LANES {
            let mut s = 0.0f32;
            for dim in 0..d {
                let diff = query[dim] - cb[base + dim * LANES + lane];
                s += diff * diff;
            }
            if s < bestd {
                bestd = s;
                best = (b * LANES + lane) as u32;
            }
        }
    }
    (best, bestd)
}
