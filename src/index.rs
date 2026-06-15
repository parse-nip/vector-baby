//! IVF-PQ index with disk-backed, memory-mapped storage.
//!
//! Layout on disk (in `dir`):
//!   meta.json      – parameters, list sizes/offsets, dataset spec
//!   centroids.bin  – nlist * d   f32   (coarse quantizer)
//!   codebook.bin   – m * ksub * dsub f32 (product quantizer)
//!   codes.bin      – n * m       u8    (PQ codes, grouped by inverted list)
//!   ids.bin        – n           u32   (original vector id per code)
//!
//! Only `centroids.bin` and `codebook.bin` are resident in RAM (a few MB).
//! `codes.bin`/`ids.bin` are mmap'd, so a billion vectors live on disk and the
//! OS pages in only what each query touches.

use crate::dataset::{Dataset, DatasetSpec};
use crate::kmeans::kmeans;
use crate::math::{argmin_blk, l2_sq, to_blocks};
use crate::pq::{adc_distance, Pq, PqParams};
use crate::topk::TopK;
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Serialize, Deserialize)]
pub struct Meta {
    pub dataset: DatasetSpec,
    pub n: u64,
    pub nlist: usize,
    pub d: usize,
    pub pq: PqParams,
    pub list_offsets: Vec<u64>, // len nlist+1
}

pub struct BuildParams {
    pub nlist: usize,
    pub m: usize,
    pub ksub: usize,
    pub n_train: usize,
    pub coarse_iters: usize,
    pub pq_iters: usize,
    pub batch: usize,
}

fn write_f32(path: &Path, data: &[f32]) -> std::io::Result<()> {
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(bytes)?;
    w.flush()
}

fn read_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let mut buf = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut buf)?;
    let n = buf.len() / 4;
    let mut out = vec![0.0f32; n];
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    dst.copy_from_slice(&buf[..n * 4]);
    Ok(out)
}

pub struct IvfPq {
    pub meta: Meta,
    pub centroids: Vec<f32>, // nlist * d
    pub pq: Pq,
    codes: Mmap, // n * m
    ids: Mmap,   // n * 4 (u32 LE)
}

impl IvfPq {
    fn p(dir: &Path, name: &str) -> PathBuf {
        dir.join(name)
    }

    /// Build the full index on disk. `progress` is called with human-readable
    /// status lines.
    pub fn build<F: Fn(String)>(
        dir: &Path,
        spec: DatasetSpec,
        n: u64,
        bp: &BuildParams,
        progress: F,
    ) -> std::io::Result<IvfPq> {
        std::fs::create_dir_all(dir)?;
        let d = spec.d;
        let m = bp.m;
        let ksub = bp.ksub;
        assert!(d % m == 0, "d must be divisible by m");
        let dsub = d / m;
        let ds = Dataset::new(spec.clone());

        // ---- Train coarse + PQ on a sample ----
        let t0 = Instant::now();
        let n_train = bp.n_train.min(n as usize);
        progress(format!("training on {} sample vectors", n_train));
        let mut sample = vec![0.0f32; n_train * d];
        ds.gen_block(0, n_train, &mut sample);

        let centroids = kmeans(&sample, n_train, d, bp.nlist, bp.coarse_iters, spec.seed ^ 0xC0FFEE);
        let cent_b = to_blocks(&centroids, bp.nlist, d); // block16 layout, fast assignment
        progress(format!("coarse k-means done ({} lists) in {:.1}s", bp.nlist, t0.elapsed().as_secs_f64()));

        // residuals of the sample for PQ training
        let mut residuals = vec![0.0f32; n_train * d];
        residuals
            .par_chunks_mut(d)
            .enumerate()
            .for_each(|(i, res)| {
                let x = &sample[i * d..i * d + d];
                let (c, _) = argmin_blk(x, &cent_b, bp.nlist, d);
                let cc = &centroids[c as usize * d..c as usize * d + d];
                for j in 0..d {
                    res[j] = x[j] - cc[j];
                }
            });
        let pq = Pq::train(&residuals, n_train, d, m, ksub, bp.pq_iters, spec.seed ^ 0xBEEF);
        drop(sample);
        drop(residuals);
        progress(format!("PQ training done in {:.1}s", t0.elapsed().as_secs_f64()));

        write_f32(&Self::p(dir, "centroids.bin"), &centroids)?;
        write_f32(&Self::p(dir, "codebook.bin"), &pq.codebook)?;

        // ---- Pass A: assign every vector to a coarse list, count sizes ----
        let nlist = bp.nlist;
        let batch = bp.batch;
        let mut list_sizes = vec![0u64; nlist];
        let listid_path = Self::p(dir, "listid.tmp");
        {
            let mut w = BufWriter::new(File::create(&listid_path)?);
            let mut vecs = vec![0.0f32; batch * d];
            let mut assign = vec![0u32; batch];
            let mut done: u64 = 0;
            let ta = Instant::now();
            while done < n {
                let cur = ((n - done) as usize).min(batch);
                ds.gen_block(done, cur, &mut vecs[..cur * d]);
                assign[..cur]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(j, a)| {
                        let x = &vecs[j * d..j * d + d];
                        *a = argmin_blk(x, &cent_b, nlist, d).0;
                    });
                for j in 0..cur {
                    list_sizes[assign[j] as usize] += 1;
                }
                let bytes = unsafe {
                    std::slice::from_raw_parts(assign.as_ptr() as *const u8, cur * 4)
                };
                w.write_all(bytes)?;
                done += cur as u64;
                if done % (batch as u64 * 20) == 0 || done == n {
                    let rate = done as f64 / ta.elapsed().as_secs_f64();
                    progress(format!(
                        "assign pass: {}/{} ({:.2}M vec/s)",
                        done, n, rate / 1e6
                    ));
                }
            }
            w.flush()?;
        }

        // ---- offsets ----
        let mut list_offsets = vec![0u64; nlist + 1];
        for l in 0..nlist {
            list_offsets[l + 1] = list_offsets[l] + list_sizes[l];
        }
        assert_eq!(list_offsets[nlist], n);

        // ---- allocate output files ----
        let codes_path = Self::p(dir, "codes.bin");
        let ids_path = Self::p(dir, "ids.bin");
        {
            let f = File::create(&codes_path)?;
            f.set_len(n * m as u64)?;
        }
        {
            let f = File::create(&ids_path)?;
            f.set_len(n * 4)?;
        }
        let codes_file = std::fs::OpenOptions::new().read(true).write(true).open(&codes_path)?;
        let ids_file = std::fs::OpenOptions::new().read(true).write(true).open(&ids_path)?;
        let mut codes_mm = unsafe { memmap2::MmapMut::map_mut(&codes_file)? };
        let mut ids_mm = unsafe { memmap2::MmapMut::map_mut(&ids_file)? };

        // ---- Pass B: encode + scatter into list-grouped layout ----
        {
            let mut rdr = BufReader::new(File::open(&listid_path)?);
            let mut cursor = list_offsets.clone(); // write position per list
            let mut vecs = vec![0.0f32; batch * d];
            let mut listid = vec![0u32; batch];
            let mut codes_batch = vec![0u8; batch * m];
            let mut done: u64 = 0;
            let tb = Instant::now();
            while done < n {
                let cur = ((n - done) as usize).min(batch);
                ds.gen_block(done, cur, &mut vecs[..cur * d]);
                let lid_bytes = unsafe {
                    std::slice::from_raw_parts_mut(listid.as_mut_ptr() as *mut u8, cur * 4)
                };
                rdr.read_exact(lid_bytes)?;

                // parallel encode residuals
                codes_batch[..cur * m]
                    .par_chunks_mut(m)
                    .enumerate()
                    .for_each(|(j, code)| {
                        let x = &vecs[j * d..j * d + d];
                        let c = listid[j] as usize;
                        let cc = &centroids[c * d..c * d + d];
                        // residual on the stack
                        let mut res = [0.0f32; 4096];
                        for k in 0..d {
                            res[k] = x[k] - cc[k];
                        }
                        pq.encode_into(&res[..d], code);
                    });

                // sequential scatter (cheap copies)
                let codes_dst = &mut codes_mm[..];
                let ids_dst = &mut ids_mm[..];
                for j in 0..cur {
                    let l = listid[j] as usize;
                    let pos = cursor[l] as usize;
                    cursor[l] += 1;
                    codes_dst[pos * m..pos * m + m].copy_from_slice(&codes_batch[j * m..j * m + m]);
                    let gid = (done + j as u64) as u32;
                    ids_dst[pos * 4..pos * 4 + 4].copy_from_slice(&gid.to_le_bytes());
                }
                done += cur as u64;
                if done % (batch as u64 * 20) == 0 || done == n {
                    let rate = done as f64 / tb.elapsed().as_secs_f64();
                    progress(format!(
                        "encode pass: {}/{} ({:.2}M vec/s)",
                        done, n, rate / 1e6
                    ));
                }
            }
            codes_mm.flush()?;
            ids_mm.flush()?;
        }
        let _ = std::fs::remove_file(&listid_path);

        let meta = Meta {
            dataset: spec,
            n,
            nlist,
            d,
            pq: PqParams { m, ksub, dsub },
            list_offsets,
        };
        let meta_json = serde_json::to_string_pretty(&meta).unwrap();
        std::fs::write(Self::p(dir, "meta.json"), meta_json)?;
        progress(format!("build complete: {} vectors in {:.1}s", n, t0.elapsed().as_secs_f64()));

        Self::open(dir)
    }

    pub fn open(dir: &Path) -> std::io::Result<IvfPq> {
        let meta: Meta = serde_json::from_slice(&std::fs::read(Self::p(dir, "meta.json"))?)?;
        let centroids = read_f32(&Self::p(dir, "centroids.bin"))?;
        let codebook = read_f32(&Self::p(dir, "codebook.bin"))?;
        let pq = Pq::from_parts(meta.pq.clone(), codebook);
        let codes = unsafe { Mmap::map(&File::open(Self::p(dir, "codes.bin"))?)? };
        let ids = unsafe { Mmap::map(&File::open(Self::p(dir, "ids.bin"))?)? };
        Ok(IvfPq { meta, centroids, pq, codes, ids })
    }

    /// Advise the OS to prefetch the mmap'd code store (best-effort warmup).
    pub fn warmup(&self) {
        // Touch one byte per page to pull pages into the page cache.
        let step = 4096;
        let mut s: u64 = 0;
        let mut i = 0;
        while i < self.codes.len() {
            s = s.wrapping_add(self.codes[i] as u64);
            i += step;
        }
        std::hint::black_box(s);
    }

    #[inline]
    fn id_at(&self, pos: usize) -> u32 {
        let b = &self.ids[pos * 4..pos * 4 + 4];
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// Search: returns up to `k` (id, approx_distance) pairs, ascending.
    pub fn search(&self, query: &[f32], nprobe: usize, k: usize) -> Vec<(u32, f32)> {
        let d = self.meta.d;
        let nlist = self.meta.nlist;
        let m = self.meta.pq.m;
        let ksub = self.meta.pq.ksub;

        // --- coarse: pick nprobe nearest lists ---
        let mut coarse = TopK::new(nprobe);
        for c in 0..nlist {
            let dist = l2_sq(query, &self.centroids[c * d..c * d + d]);
            coarse.push(dist, c as u32);
        }
        let probes = coarse.into_sorted(); // (dist, list_id)

        // --- scan probed lists in parallel, merge top-k ---
        let merged = probes
            .par_iter()
            .map(|&(_, lid)| {
                let l = lid as usize;
                let start = self.meta.list_offsets[l] as usize;
                let end = self.meta.list_offsets[l + 1] as usize;
                let cc = &self.centroids[l * d..l * d + d];
                let mut res = [0.0f32; 4096];
                for j in 0..d {
                    res[j] = query[j] - cc[j];
                }
                let mut lut = vec![0.0f32; m * ksub];
                self.pq.build_lut(&res[..d], &mut lut);

                let mut top = TopK::new(k);
                let codes = &self.codes[start * m..end * m];
                for (p, code) in codes.chunks_exact(m).enumerate() {
                    let dist = adc_distance(&lut, code, m, ksub);
                    if top.would_accept(dist) {
                        top.push(dist, (start + p) as u32);
                    }
                }
                top
            })
            .reduce(|| TopK::new(k), |mut a, b| {
                for (dist, pos) in b.iter() {
                    a.push(dist, pos);
                }
                a
            });

        merged
            .into_sorted()
            .into_iter()
            .map(|(dist, pos)| (self.id_at(pos as usize), dist))
            .collect()
    }
}
