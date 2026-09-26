//! Sparse TF-IDF inverted index over Source 2 + Source 3, queried with Source 1 records.
//!
//! Each record becomes a set of hashed tokens split into 4 channels (name words, name
//! skeletons, address words, address skeletons). Per channel the record vector is
//! idf-weighted and L2-normalised; the blocking score is a weighted sum of per-channel
//! cosines. Because every token hash includes the record's country label, postings (and
//! document frequencies) are automatically per-country, with no hard-coded country list.

use std::cell::RefCell;
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::io::Rec;
use crate::normalize::{channel, country_key, record_tokens, CHANNEL_GROUP, N_CHANNELS, N_GROUPS};

pub struct Index {
    pub ids: Vec<String>,
    /// Whether the record has any address text. Empty-address records can only be found by
    /// name, so they get their own name-ranked candidate list.
    has_addr: Vec<bool>,
    vocab: Vec<u64>,
    df: Vec<u32>,
    idf: Vec<f32>,
    post_off: Vec<u64>,
    post_doc: Vec<u32>,
    post_w: Vec<f32>,
    country_n: FxHashMap<u64, u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct Cand {
    pub doc: u32,
    /// Per-group cosine contributions (already alpha-weighted); see `CHANNEL_GROUP`.
    pub s: [f32; N_GROUPS],
}

impl Cand {
    #[inline]
    pub fn score(&self) -> f32 {
        self.s.iter().sum()
    }

    /// Name AND address agreement: high only when both sides match, so a namesake in
    /// another city (name 2.4, address 0) cannot outrank a same-address record whose name
    /// is transliterated or abbreviated.
    #[inline]
    pub fn agreement(&self) -> f32 {
        (0.5 * (self.s[0] + self.s[1] + self.s[2])).min(self.s[3])
    }
}

/// Per-thread scratch space reused across queries.
#[derive(Default)]
pub struct QueryBuf {
    /// Per-doc group scores, interleaved so one posting touches one cache line.
    scores: Vec<[f32; N_GROUPS]>,
    touched: Vec<u32>,
    tmp: Vec<u64>,
    pool: Vec<Cand>,
}

#[derive(Clone, Copy)]
pub struct QueryParams {
    pub k: usize,
    pub k_both: usize,
    pub k_noaddr: usize,
    pub k_group: [usize; N_GROUPS],
    pub max_df: u32,
    pub alpha: [f32; N_CHANNELS],
}

#[inline]
fn idf_of(n: u32, df: u32) -> f32 {
    (1.0 + n as f32 / df.max(1) as f32).ln()
}

fn tokenize_all(recs: &[Rec]) -> (Vec<u64>, Vec<u64>) {
    let chunks: Vec<(Vec<u64>, Vec<u32>)> = recs
        .par_chunks(32_768)
        .map(|ch| {
            let mut toks = Vec::with_capacity(ch.len() * 28);
            let mut lens = Vec::with_capacity(ch.len());
            for r in ch {
                let before = toks.len();
                record_tokens(r.name, r.addr, r.country, &mut toks);
                lens.push((toks.len() - before) as u32);
            }
            (toks, lens)
        })
        .collect();
    let total: usize = chunks.iter().map(|c| c.0.len()).sum();
    let mut toks = Vec::with_capacity(total);
    let mut off = Vec::with_capacity(recs.len() + 1);
    off.push(0u64);
    for (t, lens) in chunks {
        toks.extend_from_slice(&t);
        for l in lens {
            off.push(off.last().unwrap() + l as u64);
        }
    }
    (toks, off)
}

impl Index {
    pub fn build(recs: &[Rec]) -> Index {
        let t0 = Instant::now();
        let n = recs.len();
        assert!(n < u32::MAX as usize);

        let (toks, doc_off) = tokenize_all(recs);
        eprintln!("  tokenised {} docs -> {} tokens ({:.1}s)", n, toks.len(), t0.elapsed().as_secs_f32());

        let mut vocab = toks.clone();
        vocab.par_sort_unstable();
        vocab.dedup();
        vocab.shrink_to_fit();
        let tids: Vec<u32> = toks.par_iter().map(|h| vocab.binary_search(h).unwrap() as u32).collect();
        drop(toks);
        eprintln!("  vocab {} ({:.1}s)", vocab.len(), t0.elapsed().as_secs_f32());

        let v = vocab.len();
        let mut df = vec![0u32; v];
        for &t in &tids {
            df[t as usize] += 1;
        }

        let doc_country: Vec<u64> = recs.par_iter().map(|r| country_key(r.country)).collect();
        let mut country_n: FxHashMap<u64, u32> = FxHashMap::default();
        for &c in &doc_country {
            *country_n.entry(c).or_default() += 1;
        }

        // Every occurrence of a token id shares the same country (it is in the hash).
        let mut idf = vec![0f32; v];
        for d in 0..n {
            let nc = country_n[&doc_country[d]];
            for &t in &tids[doc_off[d] as usize..doc_off[d + 1] as usize] {
                idf[t as usize] = idf_of(nc, df[t as usize]);
            }
        }

        // Renumber documents for memory locality: group by country, then by the doc's two
        // most frequent address tokens (~ state / city). Posting lists of common tokens then
        // hit contiguous runs of the per-thread score array instead of random cache lines.
        let mut order: Vec<u32> = (0..n as u32).collect();
        let keys: Vec<(u64, u32, u32)> = (0..n)
            .into_par_iter()
            .map(|d| {
                let (mut a, mut b) = ((0u32, u32::MAX), (0u32, u32::MAX));
                for &t in &tids[doc_off[d] as usize..doc_off[d + 1] as usize] {
                    if CHANNEL_GROUP[channel(vocab[t as usize])] != 3 {
                        continue;
                    }
                    let f = df[t as usize];
                    if f > a.0 {
                        b = a;
                        a = (f, t);
                    } else if f > b.0 {
                        b = (f, t);
                    }
                }
                (doc_country[d], a.1, b.1)
            })
            .collect();
        order.par_sort_unstable_by_key(|&d| (keys[d as usize], d));
        drop(keys);
        let ids: Vec<String> = order.iter().map(|&d| recs[d as usize].id.to_string()).collect();
        let has_addr: Vec<bool> = order
            .par_iter()
            .map(|&d| !crate::normalize::addr_words(recs[d as usize].addr).is_empty())
            .collect();

        let mut post_off = Vec::with_capacity(v + 1);
        post_off.push(0u64);
        for &c in &df {
            post_off.push(post_off.last().unwrap() + c as u64);
        }
        let total = *post_off.last().unwrap() as usize;
        let mut post_doc = vec![0u32; total];
        let mut post_w = vec![0f32; total];
        let mut cursor: Vec<u64> = post_off[..v].to_vec();
        for (new_d, &d) in order.iter().enumerate() {
            let d = d as usize;
            let ts = &tids[doc_off[d] as usize..doc_off[d + 1] as usize];
            let mut norm = [0f32; N_CHANNELS];
            for &t in ts {
                let w = idf[t as usize];
                norm[channel(vocab[t as usize])] += w * w;
            }
            for x in norm.iter_mut() {
                *x = x.sqrt().max(1e-6);
            }
            for &t in ts {
                let p = cursor[t as usize] as usize;
                cursor[t as usize] += 1;
                post_doc[p] = new_d as u32;
                post_w[p] = idf[t as usize] / norm[channel(vocab[t as usize])];
            }
        }
        eprintln!("  postings {} ({:.1}s)", total, t0.elapsed().as_secs_f32());

        Index { ids, has_addr, vocab, df, idf, post_off, post_doc, post_w, country_n }
    }

    pub fn n_docs(&self) -> usize {
        self.ids.len()
    }

    /// Candidates for one query record, best (combined score) first: the union of the
    /// top-`k` by combined score, the top-`k_both` by name-and-address agreement, the
    /// top-`k_noaddr` by name among address-less records and, for every score group g, the top-`k_group[g]` by that
    /// group's score alone. The per-group lists stop generic names ("Orthopedic Care") or
    /// generic addresses from crowding a true match out of a single combined ranking, and
    /// let exact joined-name keys / transliterated skeleton keys retrieve on their own.
    pub fn query(&self, r: &Rec, p: &QueryParams, buf: &mut QueryBuf) -> Vec<Cand> {
        let Some(&nc) = self.country_n.get(&country_key(r.country)) else {
            return Vec::new();
        };
        buf.tmp.clear();
        record_tokens(r.name, r.addr, r.country, &mut buf.tmp);

        let mut norm = [0f32; N_CHANNELS];
        let mut terms: Vec<(Option<u32>, usize, f32)> = Vec::with_capacity(buf.tmp.len());
        for &h in buf.tmp.iter() {
            let ch = channel(h);
            let (tid, w) = match self.vocab.binary_search(&h) {
                Ok(t) => (Some(t as u32), self.idf[t]),
                Err(_) => (None, idf_of(nc, 1)),
            };
            norm[ch] += w * w;
            terms.push((tid, ch, w));
        }
        for x in norm.iter_mut() {
            *x = x.sqrt().max(1e-6);
        }

        // Traverse only selective posting lists; if none are selective, fall back to
        // the single rarest known token so no query is left without candidates.
        let mut trav: Vec<(u32, f32, usize)> = Vec::with_capacity(terms.len());
        let mut rarest: Option<(u32, u32, f32, usize)> = None;
        for &(tid, ch, w) in &terms {
            let Some(t) = tid else { continue };
            let qw = p.alpha[ch] * w / norm[ch];
            let g = CHANNEL_GROUP[ch];
            let df = self.df[t as usize];
            if df <= p.max_df {
                trav.push((t, qw, g));
            } else if rarest.map_or(true, |(_, d, _, _)| df < d) {
                rarest = Some((t, df, qw, g));
            }
        }
        if trav.is_empty() {
            if let Some((t, _, qw, g)) = rarest {
                trav.push((t, qw, g));
            }
        }

        for &(t, qw, g) in &trav {
            let a = self.post_off[t as usize] as usize;
            let b = self.post_off[t as usize + 1] as usize;
            for i in a..b {
                let d = self.post_doc[i] as usize;
                let e = &mut buf.scores[d];
                // All contributions are positive, so an all-zero slot means "not yet touched".
                if *e == [0f32; N_GROUPS] {
                    buf.touched.push(d as u32);
                }
                e[g] += qw * self.post_w[i];
            }
        }

        let pool = &mut buf.pool;
        pool.clear();
        for &d in buf.touched.iter() {
            let d = d as usize;
            let s = std::mem::take(&mut buf.scores[d]);
            pool.push(Cand { doc: d as u32, s });
        }
        buf.touched.clear();

        let mut chosen: Vec<Cand> = Vec::with_capacity(p.k + p.k_both + p.k_noaddr + p.k_group.iter().sum::<usize>());
        let mut take = |pool: &mut Vec<Cand>, k: usize, key: &dyn Fn(&Cand) -> f32| {
            if k == 0 {
                return;
            }
            let cmp = |a: &Cand, b: &Cand| key(b).total_cmp(&key(a)).then(a.doc.cmp(&b.doc));
            if pool.len() > k {
                pool.select_nth_unstable_by(k - 1, cmp);
            }
            chosen.extend(pool[..k.min(pool.len())].iter().filter(|c| key(c) > 0.0).copied());
        };
        take(pool, p.k, &|c| c.score());
        take(pool, p.k_both, &|c| c.agreement());
        let has_addr = &self.has_addr;
        take(pool, p.k_noaddr, &|c| if has_addr[c.doc as usize] { 0.0 } else { c.s[0] + c.s[1] + c.s[2] });
        for g in 0..N_GROUPS {
            take(pool, p.k_group[g], &|c| c.s[g]);
        }
        chosen.sort_unstable_by_key(|c| c.doc);
        chosen.dedup_by_key(|c| c.doc);
        chosen.sort_unstable_by(|a, b| b.score().total_cmp(&a.score()).then(a.doc.cmp(&b.doc)));
        chosen.shrink_to_fit();
        chosen
    }

    /// Locality key for a query: (country, its most frequent known address token).
    fn query_key(&self, r: &Rec) -> (u64, u32) {
        let mut tmp = Vec::new();
        record_tokens(r.name, r.addr, r.country, &mut tmp);
        let mut best = (0u32, u32::MAX);
        for h in tmp {
            if CHANNEL_GROUP[channel(h)] != 3 {
                continue;
            }
            if let Ok(t) = self.vocab.binary_search(&h) {
                if self.df[t] > best.0 {
                    best = (self.df[t], t as u32);
                }
            }
        }
        (country_key(r.country), best.1)
    }

    /// Run all queries in parallel. Queries are processed grouped by city-like key so each
    /// thread's working set of posting lists stays cache-resident; results come back in the
    /// original query order.
    pub fn query_all(&self, queries: &[Rec], p: &QueryParams) -> Vec<Vec<Cand>> {
        thread_local! {
            static BUF: RefCell<QueryBuf> = RefCell::new(QueryBuf::default());
        }
        let n = self.n_docs();
        let keys: Vec<(u64, u32)> = queries.par_iter().map(|r| self.query_key(r)).collect();
        let mut order: Vec<u32> = (0..queries.len() as u32).collect();
        order.par_sort_unstable_by_key(|&i| (keys[i as usize], i));
        let results: Vec<Vec<Cand>> = order
            .par_iter()
            .with_max_len(256)
            .map(|&i| {
                BUF.with(|b| {
                    let mut b = b.borrow_mut();
                    if b.scores.len() != n {
                        b.scores = vec![[0f32; N_GROUPS]; n];
                    }
                    self.query(&queries[i as usize], p, &mut b)
                })
            })
            .collect();
        let mut out: Vec<Vec<Cand>> = vec![Vec::new(); queries.len()];
        for (res, &i) in results.into_iter().zip(&order) {
            out[i as usize] = res;
        }
        out
    }
}
