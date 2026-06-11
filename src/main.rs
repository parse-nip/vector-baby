use std::collections::HashMap;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;
use vector_baby::dataset::{make_queries, Dataset, DatasetSpec};
use vector_baby::index::{BuildParams, IvfPq};
use vector_baby::math::{argmin_blk, to_blocks};
use vector_baby::server::{AppState, serve};
use vector_baby::{brute_force_all, qseed, QUERY_NOISE};
use rayon::prelude::*;

fn parse_args(args: &[String]) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(key) = args[i].strip_prefix("--") {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                m.insert(key.to_string(), args[i + 1].clone());
                i += 2;
            } else {
                m.insert(key.to_string(), "true".to_string());
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    m
}

fn get<T: std::str::FromStr>(m: &HashMap<String, String>, k: &str, def: T) -> T {
    m.get(k).and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn spec_from(m: &HashMap<String, String>) -> DatasetSpec {
    DatasetSpec {
        d: get(m, "dim", 128usize),
        num_centers: get(m, "centers", 200_000u64),
        center_scale: get(m, "center-scale", 1.0f32),
        noise: get(m, "noise", 0.2f32),
        seed: get(m, "seed", 88172645463325252u64),
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        eprintln!("usage: vbaby <build|bench|recall|serve> [--flags]");
        std::process::exit(1);
    }
    let cmd = argv[1].clone();
    let m = parse_args(&argv[2..]);
    let dir = m.get("dir").cloned().unwrap_or_else(|| "data/index".to_string());
    let dir = Path::new(&dir);

    match cmd.as_str() {
        "build" => {
            let spec = spec_from(&m);
            let n: u64 = get(&m, "n", 1_000_000u64);
            let bp = BuildParams {
                nlist: get(&m, "nlist", 1024usize),
                m: get(&m, "m", 8usize),
                ksub: get(&m, "ksub", 256usize),
                n_train: get(&m, "ntrain", 200_000usize),
                coarse_iters: get(&m, "coarse-iters", 10usize),
                pq_iters: get(&m, "pq-iters", 10usize),
                batch: get(&m, "batch", 1_000_000usize),
            };
            println!(
                "building index: n={} d={} nlist={} m={} ksub={} -> {}",
                n, spec.d, bp.nlist, bp.m, bp.ksub, dir.display()
            );
            IvfPq::build(dir, spec, n, &bp, |s| println!("[build] {}", s)).expect("build");
        }
        "bench" => {
            let idx = IvfPq::open(dir).expect("open index");
            let nq: usize = get(&m, "nq", 1000usize);
            let nprobe: usize = get(&m, "nprobe", 16usize);
            let k: usize = get(&m, "k", 10usize);
            let do_warmup: bool = get(&m, "warmup", true);
            let ds = Dataset::new(idx.meta.dataset.clone());
            let queries = make_queries(&ds, idx.meta.n, nq, QUERY_NOISE, qseed(idx.meta.dataset.seed));

            println!(
                "index: {} vectors, {} lists, d={}, {} bytes/vec (codes={:.1} GB)",
                idx.meta.n,
                idx.meta.nlist,
                idx.meta.d,
                idx.meta.pq.m,
                idx.meta.n as f64 * idx.meta.pq.m as f64 / 1e9
            );
            if do_warmup {
                let t = Instant::now();
                idx.warmup();
                println!("warmup (paged in code store) in {:.1}s", t.elapsed().as_secs_f64());
            }

            let mut lat = Vec::with_capacity(nq);
            let mut hits = 0usize;
            // a few warm iterations
            for j in 0..nq.min(20) {
                let _ = idx.search(queries.query(j), nprobe, k);
            }
            for j in 0..nq {
                let q = queries.query(j);
                let t = Instant::now();
                let res = idx.search(q, nprobe, k);
                lat.push(t.elapsed().as_secs_f64() * 1000.0);
                if res.iter().any(|&(id, _)| id as u64 == queries.targets[j]) {
                    hits += 1;
                }
            }
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = lat.iter().sum::<f64>() / lat.len() as f64;
            println!("--- query benchmark (nq={}, nprobe={}, k={}) ---", nq, nprobe, k);
            println!("latency ms: mean={:.2}  p50={:.2}  p90={:.2}  p99={:.2}  max={:.2}",
                mean, percentile(&lat, 50.0), percentile(&lat, 90.0), percentile(&lat, 99.0), percentile(&lat, 100.0));
            println!("throughput: {:.0} queries/sec (single-query latency)", 1000.0 / mean);
            println!("planted recall@{}: {:.2}% ({}/{} queries found their true NN)",
                k, 100.0 * hits as f64 / nq as f64, hits, nq);
            println!("under 100ms: {}", percentile(&lat, 99.0) < 100.0);
        }
        "recall" => {
            // Exact brute-force recall. Only run at small n (it scans everything).
            let idx = IvfPq::open(dir).expect("open index");
            let nq: usize = get(&m, "nq", 1000usize);
            let nprobe: usize = get(&m, "nprobe", 16usize);
            let k: usize = get(&m, "k", 10usize);
            let ds = Dataset::new(idx.meta.dataset.clone());
            let queries = make_queries(&ds, idx.meta.n, nq, QUERY_NOISE, qseed(idx.meta.dataset.seed));

            println!("brute-force ground truth over {} vectors...", idx.meta.n);
            let t = Instant::now();
            let truth = brute_force_all(&ds, idx.meta.n, &queries, k, 1_000_000);
            println!("brute force done in {:.1}s", t.elapsed().as_secs_f64());

            let mut recall_sum = 0.0f64;
            let mut planted = 0usize;
            for j in 0..nq {
                let res = idx.search(queries.query(j), nprobe, k);
                let got: std::collections::HashSet<u32> = res.iter().map(|&(id, _)| id).collect();
                let truth_ids: std::collections::HashSet<u32> =
                    truth[j].iter().map(|&(_, id)| id).collect();
                let inter = got.intersection(&truth_ids).count();
                recall_sum += inter as f64 / k as f64;
                if got.contains(&(queries.targets[j] as u32)) {
                    planted += 1;
                }
            }
            println!("--- recall vs exact brute force (nq={}, nprobe={}, k={}) ---", nq, nprobe, k);
            println!("recall@{} = {:.2}%", k, 100.0 * recall_sum / nq as f64);
            println!("planted recall@{} = {:.2}%", k, 100.0 * planted as f64 / nq as f64);
        }
        "diag" => {
            // Isolate each build stage to attribute the bottleneck.
            let idx = IvfPq::open(dir).expect("open index (build a small one first)");
            let d = idx.meta.d;
            let nlist = idx.meta.nlist;
            let mcode = idx.meta.pq.m;
            let n: u64 = get(&m, "n", 4_000_000u64);
            let batch: usize = get(&m, "batch", 1_000_000usize);
            let ds = Dataset::new(idx.meta.dataset.clone());
            let cent_b = to_blocks(&idx.centroids, nlist, d);
            let threads = rayon::current_num_threads();
            println!("threads={} d={} nlist={} m={}", threads, d, nlist, mcode);

            // --- (A) single-thread raw kernel throughput ---
            let q = 8192usize;
            let mut qbuf = vec![0.0f32; q * d];
            ds.gen_block(0, q, &mut qbuf);
            let reps = 256usize;
            let t = Instant::now();
            let mut acc = 0u64;
            for _ in 0..reps {
                for j in 0..q {
                    let (bi, _) = argmin_blk(&qbuf[j * d..j * d + d], &cent_b, nlist, d);
                    acc = acc.wrapping_add(bi as u64);
                }
            }
            black_box(acc);
            let secs = t.elapsed().as_secs_f64();
            let assigns = (q * reps) as f64;
            let gflops = assigns * nlist as f64 * d as f64 * 2.0 / secs / 1e9;
            println!(
                "[A] kernel 1-thread: {:.2}M assign/s, {:.1} GFLOP/s  (1 assign = {} dist of dim {})",
                assigns / secs / 1e6,
                gflops,
                nlist,
                d
            );

            // --- (B) generation only (parallel) ---
            let mut vecs = vec![0.0f32; batch * d];
            let mut done = 0u64;
            let t = Instant::now();
            let mut cksum = 0.0f32;
            while done < n {
                let cur = ((n - done) as usize).min(batch);
                ds.gen_block(done, cur, &mut vecs[..cur * d]);
                cksum += vecs[0];
                done += cur as u64;
            }
            black_box(cksum);
            let t_gen = t.elapsed().as_secs_f64();
            println!("[B] generate only:        {:.3}M vec/s", n as f64 / t_gen / 1e6);

            // --- (C) generate + assign (parallel) ---
            let mut assign = vec![0u32; batch];
            let mut done = 0u64;
            let t = Instant::now();
            while done < n {
                let cur = ((n - done) as usize).min(batch);
                ds.gen_block(done, cur, &mut vecs[..cur * d]);
                assign[..cur].par_iter_mut().enumerate().for_each(|(j, a)| {
                    *a = argmin_blk(&vecs[j * d..j * d + d], &cent_b, nlist, d).0;
                });
                done += cur as u64;
            }
            black_box(assign[0]);
            let t_assign = t.elapsed().as_secs_f64();
            println!("[C] generate + assign:    {:.3}M vec/s", n as f64 / t_assign / 1e6);

            // --- (D) generate + assign + residual + encode (full pass) ---
            let mut codes = vec![0u8; batch * mcode];
            let mut done = 0u64;
            let t = Instant::now();
            while done < n {
                let cur = ((n - done) as usize).min(batch);
                ds.gen_block(done, cur, &mut vecs[..cur * d]);
                assign[..cur].par_iter_mut().enumerate().for_each(|(j, a)| {
                    *a = argmin_blk(&vecs[j * d..j * d + d], &cent_b, nlist, d).0;
                });
                codes[..cur * mcode].par_chunks_mut(mcode).enumerate().for_each(|(j, code)| {
                    let c = assign[j] as usize;
                    let cc = &idx.centroids[c * d..c * d + d];
                    let mut res = [0.0f32; 4096];
                    for k in 0..d {
                        res[k] = vecs[j * d + k] - cc[k];
                    }
                    idx.pq.encode_into(&res[..d], code);
                });
                done += cur as u64;
            }
            black_box(codes[0]);
            let t_full = t.elapsed().as_secs_f64();
            println!("[D] gen+assign+encode:    {:.3}M vec/s", n as f64 / t_full / 1e6);

            // --- attribution + 1B projection ---
            let per = |secs: f64| secs / n as f64;
            let gen_s = per(t_gen);
            let assign_s = (per(t_assign) - per(t_gen)).max(0.0);
            let encode_s = (per(t_full) - per(t_assign)).max(0.0);
            println!("--- per-vector cost (parallel, {} threads) ---", threads);
            println!("  generate: {:.1} ns   assign: {:.1} ns   encode: {:.1} ns",
                gen_s * 1e9, assign_s * 1e9, encode_s * 1e9);
            println!("--- projected time to build 1,000,000,000 vectors ---");
            println!("  assign pass (gen+assign):        {:.1} min", per(t_assign) * 1e9 / 60.0);
            println!("  encode pass (gen+encode approx): {:.1} min", (gen_s + encode_s) * 1e9 / 60.0);
            println!("  TOTAL (two passes):              {:.1} min", (per(t_assign) + gen_s + encode_s) * 1e9 / 60.0);
        }
        "serve" => {
            let idx = IvfPq::open(dir).expect("open index");
            let port: u16 = get(&m, "port", 8080u16);
            let nq: usize = get(&m, "nq", 256usize);
            let nprobe: usize = get(&m, "nprobe", 16usize);
            let k: usize = get(&m, "k", 10usize);
            let ds = Dataset::new(idx.meta.dataset.clone());
            let queries = make_queries(&ds, idx.meta.n, nq, QUERY_NOISE, qseed(idx.meta.dataset.seed));
            println!("warming up code store...");
            idx.warmup();
            let state = AppState {
                index: std::sync::Arc::new(idx),
                ds,
                queries,
                nprobe,
                k,
            };
            serve(state, port);
        }
        other => {
            eprintln!("unknown command: {}", other);
            std::process::exit(1);
        }
    }
}
