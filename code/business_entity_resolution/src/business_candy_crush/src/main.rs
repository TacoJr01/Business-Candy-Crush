//! business_candy_crush — scalable blocking / candidate generation for the
//! ML Challenge 2026 business entity resolution task (Part 1).
//!
//!   business_candy_crush block --s1 S1.tsv --s2 S2.tsv --s3 S3.tsv --out-dir DIR
//!       [--k 15] [--k-both 10] [--k-noaddr 5] [--k-group 5,10,10,15] [--max-df 100000] [--alpha 1,0.5,0.5,1,0.3,0.5]
//!       [--tau 1.6] [--val-only] [--limit N] [--jobs N]
//!   k-group order: name words, joined-name key, fuzzy name, address.
//!   alpha order: name words, joined-name key, fuzzy name, address, address skeleton, name trigrams.
//!   business_candy_crush eval --scored DIR/candidates_scored.tsv --gt GT.tsv --s1 S1.tsv
//!       [--misses DIR/misses.tsv]

mod eval;
mod index;
mod io;
mod normalize;

use std::time::Instant;

use index::{Index, QueryParams};
use io::{parse_source, read_to_string, write_lists};
use normalize::fnv1a64;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn req(args: &[String], name: &str) -> String {
    arg(args, name).unwrap_or_else(|| {
        eprintln!("missing required argument {name}");
        std::process::exit(2)
    })
}

/// Cap rayon at ~80% of logical CPUs (CPU/RAM restraint). Precedence:
/// `--jobs N` > `RAYON_NUM_THREADS` > 80% of available parallelism.
/// An explicit value is honoured as-is (user wish); the default is the cap.
/// `PART2_ALLOW_FULL=1` lifts the cap for explicit values (never for the default).
fn init_thread_pool(args: &[String]) {
    let sys = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let cap = ((sys as f64 * 0.8).floor() as usize).max(1);
    let allow_full = std::env::var("PART2_ALLOW_FULL").map(|v| v == "1" || v == "true").unwrap_or(false);
    let want: Option<usize> = arg(args, "--jobs")
        .and_then(|x| x.parse().ok())
        .or_else(|| std::env::var("RAYON_NUM_THREADS").ok().and_then(|x| x.parse().ok()));
    let n = match want {
        Some(w) if allow_full => w.max(1),
        Some(w) => w.max(1).min(cap.max(1)),
        None => cap,
    };
    let n = n.min(sys).max(1);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    eprintln!("threads: using {n}/{sys} (80% cap = {cap})");
}

fn block(args: &[String]) {
    let t0 = Instant::now();
    let out_dir = req(args, "--out-dir");
    std::fs::create_dir_all(&out_dir).unwrap();
    let floats = |name: &str, default: &str| -> Vec<f32> {
        arg(args, name).unwrap_or_else(|| default.into()).split(',').map(|x| x.parse().unwrap()).collect()
    };
    let alpha = floats("--alpha", "1,0.5,0.5,1,0.3,0.5");
    let kg: Vec<usize> = floats("--k-group", "5,10,10,15").iter().map(|&x| x as usize).collect();
    let p = QueryParams {
        k: arg(args, "--k").map_or(15, |x| x.parse().unwrap()),
        k_both: arg(args, "--k-both").map_or(10, |x| x.parse().unwrap()),
        k_noaddr: arg(args, "--k-noaddr").map_or(5, |x| x.parse().unwrap()),
        k_group: [kg[0], kg[1], kg[2], kg[3]],
        max_df: arg(args, "--max-df").map_or(100_000, |x| x.parse().unwrap()),
        alpha: [alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]],
    };
    let tau: f32 = arg(args, "--tau").map_or(1.6, |x| x.parse().unwrap());
    let val_only = args.iter().any(|a| a == "--val-only");

    let (b2, b3) = rayon::join(|| read_to_string(&req(args, "--s2")), || read_to_string(&req(args, "--s3")));
    let mut docs = parse_source(&b2);
    docs.extend(parse_source(&b3));
    eprintln!("read {} S2+S3 records ({:.1}s)", docs.len(), t0.elapsed().as_secs_f32());

    let index = Index::build(&docs);
    drop(docs);

    let b1 = read_to_string(&req(args, "--s1"));
    let mut queries = parse_source(&b1);
    if val_only {
        queries.retain(|r| fnv1a64(r.id) % 10 == 0);
    }
    if let Some(n) = arg(args, "--limit") {
        queries.truncate(n.parse().unwrap());
    }
    eprintln!(
        "querying {} S1 records (k={}, k_both={}, k_noaddr={}, k_group={:?}, max_df={}, alpha={:?})",
        queries.len(), p.k, p.k_both, p.k_noaddr, p.k_group, p.max_df, p.alpha
    );
    let tq = Instant::now();
    let results = index.query_all(&queries, &p);
    let n_c: usize = results.iter().map(Vec::len).sum();
    eprintln!(
        "  done in {:.1}s ({:.0} q/s), avg {:.2} candidates / S1",
        tq.elapsed().as_secs_f32(),
        queries.len() as f32 / tq.elapsed().as_secs_f32(),
        n_c as f64 / queries.len() as f64
    );

    let od = |f: &str| format!("{out_dir}/{f}");
    write_lists(&od("candidate_pairs.tsv"), "candidate_entity_ids", &queries, &results, &index.ids, f32::MIN, false);
    write_lists(&od("candidates_scored.tsv"), "candidate_entity_ids", &queries, &results, &index.ids, f32::MIN, true);
    write_lists(&od("matching_results.tsv"), "matched_entity_ids", &queries, &results, &index.ids, tau, false);
    eprintln!("wrote outputs to {out_dir} (total {:.1}s)", t0.elapsed().as_secs_f32());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    init_thread_pool(&args);
    match args.get(1).map(String::as_str) {
        Some("block") => block(&args),
        Some("eval") => eval::run(
            &req(&args, "--scored"),
            &req(&args, "--gt"),
            &req(&args, "--s1"),
            arg(&args, "--misses").as_deref(),
        ),
        _ => {
            eprintln!("usage: business_candy_crush <block|eval> ... (see src/main.rs)");
            std::process::exit(2);
        }
    }
}
