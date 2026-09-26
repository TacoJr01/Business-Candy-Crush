//! TSV reading / writing. All files are tab-separated with a header row.

use std::fs::File;
use std::io::{BufWriter, Write};

use crate::index::Cand;

pub struct Rec<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub addr: &'a str,
    pub country: &'a str,
}

pub fn read_to_string(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

/// Parse a source file (entity_id, business_name, business_address, country).
pub fn parse_source(buf: &str) -> Vec<Rec<'_>> {
    buf.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split('\t');
            let id = it.next().unwrap_or("").trim();
            let name = it.next().unwrap_or("");
            let addr = it.next().unwrap_or("");
            let country = it.next().unwrap_or("");
            Rec { id, name, addr, country }
        })
        .collect()
}

/// Write one row per query: `s1_id \t id,id,...` (or `id:total:name:join:fuzzy:addr,...` when `with_scores`).
/// Only candidates with score >= `min_score` are written.
pub fn write_lists(
    path: &str,
    header: &str,
    queries: &[Rec],
    results: &[Vec<Cand>],
    ids: &[String],
    min_score: f32,
    with_scores: bool,
) {
    let f = File::create(path).unwrap_or_else(|e| panic!("cannot create {path}: {e}"));
    let mut w = BufWriter::with_capacity(1 << 22, f);
    writeln!(w, "source1_entity_id\t{header}").unwrap();
    for (q, res) in queries.iter().zip(results) {
        w.write_all(q.id.as_bytes()).unwrap();
        w.write_all(b"\t").unwrap();
        let mut first = true;
        for c in res {
            if c.score() < min_score {
                continue;
            }
            if !first {
                w.write_all(b",").unwrap();
            }
            first = false;
            w.write_all(ids[c.doc as usize].as_bytes()).unwrap();
            if with_scores {
                write!(w, ":{:.4}", c.score()).unwrap();
                for g in c.s {
                    write!(w, ":{g:.4}").unwrap();
                }
            }
        }
        w.write_all(b"\n").unwrap();
    }
    w.flush().unwrap();
}
