//! Blocking evaluation on the training holdout: recall@k, oracle F0.5 ceiling, avg
//! candidates per S1, and a naive score-threshold matcher as the baseline Part 2 must beat.

use std::fs::File;
use std::io::{BufWriter, Write};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::io::{parse_source, read_to_string};

pub fn f05(tp: usize, n_pred: usize, n_true: usize) -> f64 {
    if n_true == 0 {
        return if n_pred == 0 { 1.0 } else { 0.0 };
    }
    if tp == 0 {
        return 0.0;
    }
    let p = tp as f64 / n_pred as f64;
    let r = tp as f64 / n_true as f64;
    1.25 * p * r / (0.25 * p + r)
}

struct Row {
    country: String,
    cands: Vec<(String, f32)>,
    truth: FxHashSet<String>,
}

pub fn run(scored: &str, gt: &str, s1: &str, misses_path: Option<&str>) {
    let scored_buf = read_to_string(scored);
    let mut rows: FxHashMap<&str, Row> = FxHashMap::default();
    for l in scored_buf.lines().skip(1) {
        let (id, rest) = l.split_once('\t').unwrap_or((l, ""));
        let cands = rest
            .split(',')
            .filter(|x| !x.is_empty())
            .map(|x| {
                let mut it = x.split(':');
                let c = it.next().unwrap();
                let s = it.next().expect("scored file needs id:score");
                (c.to_string(), s.parse::<f32>().unwrap())
            })
            .collect();
        rows.insert(id, Row { country: String::new(), cands, truth: FxHashSet::default() });
    }

    let gt_buf = read_to_string(gt);
    for l in gt_buf.lines().skip(1) {
        let (id, rest) = l.split_once('\t').unwrap_or((l, ""));
        if let Some(r) = rows.get_mut(id) {
            r.truth = rest.split(',').filter(|x| !x.is_empty()).map(str::to_string).collect();
        }
    }
    let s1_buf = read_to_string(s1);
    for r in parse_source(&s1_buf) {
        if let Some(row) = rows.get_mut(r.id) {
            row.country = r.country.to_string();
        }
    }

    let kmax = rows.values().map(|r| r.cands.len()).max().unwrap_or(0);
    let mut ks: Vec<usize> = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
        .into_iter()
        .filter(|&k| k <= kmax)
        .collect();
    if ks.last() != Some(&kmax) && kmax > 0 {
        ks.push(kmax);
    }

    let mut countries: Vec<String> = rows.values().map(|r| r.country.clone()).collect();
    countries.sort();
    countries.dedup();
    countries.insert(0, "ALL".to_string());

    let n_single = rows.values().filter(|r| r.truth.is_empty()).count();
    let n_pairs: usize = rows.values().map(|r| r.truth.len()).sum();
    println!(
        "holdout S1 entities: {}  (singletons {:.1}%)  true pairs: {}",
        rows.len(),
        100.0 * n_single as f64 / rows.len() as f64,
        n_pairs
    );

    for c in &countries {
        let sel: Vec<&Row> = rows.values().filter(|r| c == "ALL" || &r.country == c).collect();
        let pairs: usize = sel.iter().map(|r| r.truth.len()).sum();
        let nonsingle = sel.iter().filter(|r| !r.truth.is_empty()).count();
        println!("\n== {c}  ({} S1, {} true pairs)", sel.len(), pairs);
        println!("{:>5} {:>11} {:>12} {:>13} {:>11}", "k", "pair_recall", "full_recall", "oracle_F0.5", "avg_cands");
        for &k in &ks {
            let mut hit = 0usize;
            let mut full = 0usize;
            let mut oracle = 0f64;
            let mut ncand = 0usize;
            for r in &sel {
                let top = &r.cands[..k.min(r.cands.len())];
                ncand += top.len();
                let tp = top.iter().filter(|(id, _)| r.truth.contains(id)).count();
                hit += tp;
                if !r.truth.is_empty() && tp == r.truth.len() {
                    full += 1;
                }
                oracle += f05(tp, tp, r.truth.len());
            }
            println!(
                "{:>5} {:>11.4} {:>12.4} {:>13.4} {:>11.2}",
                k,
                hit as f64 / pairs.max(1) as f64,
                full as f64 / nonsingle.max(1) as f64,
                oracle / sel.len() as f64,
                ncand as f64 / sel.len() as f64
            );
        }
    }

    // Naive baseline matcher: accept every candidate whose blocking score >= tau.
    println!("\n== naive threshold matcher (score >= tau), macro F0.5 on holdout");
    println!("{:>6} {:>8} {:>10} {:>10}", "tau", "F0.5", "precision", "recall");
    let mut best = (0f32, 0f64);
    for i in 0..=30 {
        let tau = 0.6 + 0.08 * i as f32;
        let (mut f, mut tp_all, mut pred_all) = (0f64, 0usize, 0usize);
        for r in rows.values() {
            let pred: Vec<&String> = r.cands.iter().filter(|(_, s)| *s >= tau).map(|(id, _)| id).collect();
            let tp = pred.iter().filter(|id| r.truth.contains(**id)).count();
            tp_all += tp;
            pred_all += pred.len();
            f += f05(tp, pred.len(), r.truth.len());
        }
        let f = f / rows.len() as f64;
        if f > best.1 {
            best = (tau, f);
        }
        println!(
            "{:>6.2} {:>8.4} {:>10.4} {:>10.4}",
            tau,
            f,
            tp_all as f64 / pred_all.max(1) as f64,
            tp_all as f64 / n_pairs.max(1) as f64
        );
    }
    println!("best tau {:.2} -> macro F0.5 {:.4}", best.0, best.1);

    if let Some(path) = misses_path {
        let mut w = BufWriter::new(File::create(path).unwrap());
        writeln!(w, "source1_entity_id\tmissed_entity_id\tn_true\tn_cands").unwrap();
        let mut n = 0;
        let mut keys: Vec<&&str> = rows.keys().collect();
        keys.sort();
        for id in keys {
            let r = &rows[*id];
            let got: FxHashSet<&str> = r.cands.iter().map(|(c, _)| c.as_str()).collect();
            for t in &r.truth {
                if !got.contains(t.as_str()) && n < 5000 {
                    writeln!(w, "{id}\t{t}\t{}\t{}", r.truth.len(), r.cands.len()).unwrap();
                    n += 1;
                }
            }
        }
        eprintln!("wrote {n} missed pairs to {path}");
    }
}
