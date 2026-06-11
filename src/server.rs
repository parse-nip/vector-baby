//! Minimal HTTP server + single-page UI for live demos. No frontend framework,
//! no build step: one embedded HTML page that calls a tiny JSON API.

use crate::dataset::{make_queries, Dataset, QuerySet};
use crate::index::IvfPq;
use crate::QUERY_NOISE;
use std::sync::Arc;
use std::time::Instant;
use tiny_http::{Header, Method, Response, Server};

const PAGE: &str = include_str!("ui.html");

pub struct AppState {
    pub index: Arc<IvfPq>,
    pub ds: Dataset,
    pub queries: QuerySet,
    pub nprobe: usize,
    pub k: usize,
}

fn parse_query(url: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    if let Some(qpos) = url.find('?') {
        for kv in url[qpos + 1..].split('&') {
            let mut it = kv.splitn(2, '=');
            if let (Some(k), Some(v)) = (it.next(), it.next()) {
                map.insert(k.to_string(), v.to_string());
            }
        }
    }
    map
}

fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap()
}
fn html_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap()
}

pub fn serve(state: AppState, port: u16) {
    let addr = format!("0.0.0.0:{}", port);
    let server = Server::http(&addr).expect("bind");
    println!("vector-baby serving on http://{}", addr);
    let state = Arc::new(state);
    let m = state.index.meta.pq.m;

    // Use a small worker pool so concurrent requests are handled.
    let pool = std::sync::Arc::new(server);
    let mut handles = Vec::new();
    for _ in 0..4 {
        let server = pool.clone();
        let state = state.clone();
        handles.push(std::thread::spawn(move || loop {
            let request = match server.recv() {
                Ok(r) => r,
                Err(_) => break,
            };
            let url = request.url().to_string();
            let method = request.method().clone();

            if url == "/" || url.starts_with("/?") {
                let _ = request.respond(Response::from_string(PAGE).with_header(html_header()));
                continue;
            }
            if url.starts_with("/api/info") {
                let meta = &state.index.meta;
                let body = format!(
                    "{{\"n\":{},\"nlist\":{},\"d\":{},\"m\":{},\"ksub\":{},\"bytes_per_vec\":{},\"nq\":{},\"nprobe\":{},\"k\":{}}}",
                    meta.n, meta.nlist, meta.d, meta.pq.m, meta.pq.ksub, m, state.queries.nq(), state.nprobe, state.k
                );
                let _ = request.respond(Response::from_string(body).with_header(json_header()));
                continue;
            }
            if url.starts_with("/api/search") && method == Method::Get {
                let params = parse_query(&url);
                let nprobe = params
                    .get("nprobe")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(state.nprobe);
                let k = params.get("k").and_then(|s| s.parse().ok()).unwrap_or(state.k);
                let nq = state.queries.nq();
                let j = match params.get("j").and_then(|s| s.parse::<usize>().ok()) {
                    Some(v) => v % nq,
                    None => {
                        // pseudo-random based on time
                        (Instant::now().elapsed().as_nanos() as usize
                            ^ std::process::id() as usize
                            ^ rand_seed()) % nq
                    }
                };
                let q = state.queries.query(j);
                let target = state.queries.targets[j];

                let t0 = Instant::now();
                let res = state.index.search(q, nprobe, k);
                let micros = t0.elapsed().as_micros();

                let hit = res.iter().any(|&(id, _)| id as u64 == target);
                let mut items = String::new();
                for (rank, (id, dist)) in res.iter().enumerate() {
                    if rank > 0 {
                        items.push(',');
                    }
                    items.push_str(&format!(
                        "{{\"rank\":{},\"id\":{},\"dist\":{:.4},\"is_target\":{}}}",
                        rank + 1,
                        id,
                        dist,
                        (*id as u64 == target)
                    ));
                }
                let body = format!(
                    "{{\"query_index\":{},\"target_id\":{},\"latency_ms\":{:.3},\"nprobe\":{},\"k\":{},\"hit\":{},\"results\":[{}]}}",
                    j, target, micros as f64 / 1000.0, nprobe, k, hit, items
                );
                let _ = request.respond(Response::from_string(body).with_header(json_header()));
                continue;
            }
            let _ = request.respond(Response::from_string("not found").with_status_code(404));
        }));
    }
    for h in handles {
        let _ = h.join();
    }
}

fn rand_seed() -> usize {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as usize)
        .unwrap_or(0)
}

pub fn build_queries(ds: &Dataset, n: u64, nq: usize, seed: u64) -> QuerySet {
    make_queries(ds, n, nq, QUERY_NOISE, seed)
}
