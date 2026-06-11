//! Distance kernels. The L2 kernel uses explicit AVX-512 when available
//! (it is on this host under `target-cpu=native`); otherwise a portable
//! scalar fallback that the autovectorizer can still handle.

#[cfg(target_feature = "avx512f")]
#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    unsafe { l2_sq_avx512(a, b) }
}

#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_sq_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0usize;
    // Two independent accumulators hide FMA latency.
    while i + 32 <= n {
        let da = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)));
        acc0 = _mm512_fmadd_ps(da, da, acc0);
        let db = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i + 16)), _mm512_loadu_ps(pb.add(i + 16)));
        acc1 = _mm512_fmadd_ps(db, db, acc1);
        i += 32;
    }
    while i + 16 <= n {
        let da = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)));
        acc0 = _mm512_fmadd_ps(da, da, acc0);
        i += 16;
    }
    let mut s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    while i < n {
        let d = *pa.add(i) - *pb.add(i);
        s += d * d;
        i += 1;
    }
    s
}

#[cfg(not(target_feature = "avx512f"))]
#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 16];
    let chunks = a.len() / 16;
    for c in 0..chunks {
        let base = c * 16;
        for j in 0..16 {
            let diff = a[base + j] - b[base + j];
            acc[j] += diff * diff;
        }
    }
    let mut s = 0.0f32;
    for j in 0..16 {
        s += acc[j];
    }
    for k in (chunks * 16)..a.len() {
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

/// Re-pack row-major centroids (`k x d`, k a multiple of 16) into a
/// block-interleaved layout: 16 centroids per block, and within a block the
/// values are stored `[dim][lane]`. So `cb[b*d*16 + dim*16 + lane]` holds
/// dimension `dim` of centroid `b*16 + lane`.
///
/// This gives the across-centroids distance kernel fully contiguous loads
/// (one cache line per dim) while still vectorizing 16 centroids at a time —
/// good ILP *and* good locality, unlike a plain transpose (whose 4 KB page
/// stride per dimension wrecks the TLB).
pub fn to_block16(src: &[f32], k: usize, d: usize) -> Vec<f32> {
    assert!(k % 16 == 0, "k must be a multiple of 16");
    let mut out = vec![0.0f32; k * d];
    for c in 0..k {
        let b = c / 16;
        let lane = c % 16;
        for dim in 0..d {
            out[b * d * 16 + dim * 16 + lane] = src[c * d + dim];
        }
    }
    out
}

/// Nearest centroid over the block-interleaved layout from [`to_block16`].
/// Returns (centroid_index, dist_sq).
#[cfg(target_feature = "avx512f")]
#[inline]
pub fn argmin_blk(query: &[f32], cb: &[f32], k: usize, d: usize) -> (u32, f32) {
    unsafe { argmin_blk_avx512(query, cb, k, d) }
}

#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f")]
unsafe fn argmin_blk_avx512(query: &[f32], cb: &[f32], k: usize, d: usize) -> (u32, f32) {
    use std::arch::x86_64::*;
    let mut best: u32 = 0;
    let mut bestd = f32::INFINITY;
    let qp = query.as_ptr();
    let mut tmp = [0.0f32; 16];
    let nblk = k / 16;

    for b in 0..nblk {
        let base = cb.as_ptr().add(b * d * 16);
        // 4 independent accumulator chains to hide FMA latency.
        let mut a0 = _mm512_setzero_ps();
        let mut a1 = _mm512_setzero_ps();
        let mut a2 = _mm512_setzero_ps();
        let mut a3 = _mm512_setzero_ps();
        let mut dim = 0usize;
        while dim + 4 <= d {
            let q0 = _mm512_set1_ps(*qp.add(dim));
            let e0 = _mm512_sub_ps(q0, _mm512_loadu_ps(base.add(dim * 16)));
            a0 = _mm512_fmadd_ps(e0, e0, a0);
            let q1 = _mm512_set1_ps(*qp.add(dim + 1));
            let e1 = _mm512_sub_ps(q1, _mm512_loadu_ps(base.add((dim + 1) * 16)));
            a1 = _mm512_fmadd_ps(e1, e1, a1);
            let q2 = _mm512_set1_ps(*qp.add(dim + 2));
            let e2 = _mm512_sub_ps(q2, _mm512_loadu_ps(base.add((dim + 2) * 16)));
            a2 = _mm512_fmadd_ps(e2, e2, a2);
            let q3 = _mm512_set1_ps(*qp.add(dim + 3));
            let e3 = _mm512_sub_ps(q3, _mm512_loadu_ps(base.add((dim + 3) * 16)));
            a3 = _mm512_fmadd_ps(e3, e3, a3);
            dim += 4;
        }
        while dim < d {
            let q0 = _mm512_set1_ps(*qp.add(dim));
            let e0 = _mm512_sub_ps(q0, _mm512_loadu_ps(base.add(dim * 16)));
            a0 = _mm512_fmadd_ps(e0, e0, a0);
            dim += 1;
        }
        let acc = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
        _mm512_storeu_ps(tmp.as_mut_ptr(), acc);
        let off = b * 16;
        for l in 0..16 {
            if tmp[l] < bestd {
                bestd = tmp[l];
                best = (off + l) as u32;
            }
        }
    }
    (best, bestd)
}

#[cfg(not(target_feature = "avx512f"))]
#[inline]
pub fn argmin_blk(query: &[f32], cb: &[f32], k: usize, d: usize) -> (u32, f32) {
    let mut best = 0u32;
    let mut bestd = f32::INFINITY;
    let nblk = k / 16;
    for b in 0..nblk {
        let base = b * d * 16;
        for lane in 0..16 {
            let mut s = 0.0f32;
            for dim in 0..d {
                let diff = query[dim] - cb[base + dim * 16 + lane];
                s += diff * diff;
            }
            if s < bestd {
                bestd = s;
                best = (b * 16 + lane) as u32;
            }
        }
    }
    (best, bestd)
}
