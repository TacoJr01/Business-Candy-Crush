//! Text normalisation and blocking-key generation.
//!
//! Everything here is country-agnostic: the country label only enters the pipeline as an
//! opaque string that is hashed into every token (so the index is implicitly partitioned
//! by whatever country labels exist, including ones never seen in training).

use std::hash::Hasher;

use rustc_hash::FxHasher;
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

/// Token channels. Encoded in the low 3 bits of every token hash.
pub const CH_NAME: u64 = 0;
pub const CH_NAME_JOIN: u64 = 1;
pub const CH_NAME_SKEL: u64 = 2;
pub const CH_ADDR: u64 = 3;
pub const CH_ADDR_SKEL: u64 = 4;
/// Character trigrams of core name words: robust to compounds glued without
/// spaces ("capitalfund" ~ "capital fund"), digit-letter swaps ("6rand" ~
/// "grand") and multi-typo words where whole-word skeleton/prefix keys drift.
pub const CH_TRIG: u64 = 5;
pub const N_CHANNELS: usize = 6;

/// Score groups: each gets its own top-k candidate list (see `Index::query`).
/// name words | joined-name key | fuzzy name (skeleton, prefix, suffix, trigrams) | address (+ skeleton)
pub const N_GROUPS: usize = 4;
pub const CHANNEL_GROUP: [usize; N_CHANNELS] = [0, 1, 2, 3, 3, 2];

// ---------------------------------------------------------------------------------------
// Indic -> Latin transliteration.
//
// All nine major Indic script blocks (Devanagari U+0900 .. Malayalam U+0D7F) share the
// ISCII-derived layout, so one table indexed by (codepoint & 0x7F) covers Hindi, Marathi,
// Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada and Malayalam.
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum K {
    Cons,
    Vowel,
    Matra,
    Virama,
    Nukta,
    Other,
}

const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

fn indic(off: u32) -> (&'static str, K) {
    use K::*;
    match off {
        0x01 | 0x02 => ("n", Other),
        0x03 => ("h", Other),
        0x05 | 0x06 => ("a", Vowel),
        0x07 | 0x08 => ("i", Vowel),
        0x09 | 0x0A => ("u", Vowel),
        0x0B | 0x60 => ("ri", Vowel),
        0x0C | 0x61 => ("li", Vowel),
        0x0D..=0x0F => ("e", Vowel),
        0x10 => ("ai", Vowel),
        0x11..=0x13 => ("o", Vowel),
        0x14 => ("au", Vowel),
        0x15 => ("k", Cons),
        0x16 => ("kh", Cons),
        0x17 => ("g", Cons),
        0x18 => ("gh", Cons),
        0x19 => ("n", Cons),
        0x1A => ("ch", Cons),
        0x1B => ("chh", Cons),
        0x1C => ("j", Cons),
        0x1D => ("jh", Cons),
        0x1E => ("n", Cons),
        0x1F => ("t", Cons),
        0x20 => ("th", Cons),
        0x21 => ("d", Cons),
        0x22 => ("dh", Cons),
        0x23 => ("n", Cons),
        0x24 => ("t", Cons),
        0x25 => ("th", Cons),
        0x26 => ("d", Cons),
        0x27 => ("dh", Cons),
        0x28 | 0x29 => ("n", Cons),
        0x2A => ("p", Cons),
        0x2B => ("ph", Cons),
        0x2C => ("b", Cons),
        0x2D => ("bh", Cons),
        0x2E => ("m", Cons),
        0x2F => ("y", Cons),
        0x30 | 0x31 => ("r", Cons),
        0x32..=0x34 => ("l", Cons),
        0x35 => ("v", Cons),
        0x36 | 0x37 => ("sh", Cons),
        0x38 => ("s", Cons),
        0x39 => ("h", Cons),
        0x58 => ("q", Cons),
        0x59 => ("kh", Cons),
        0x5A => ("g", Cons),
        0x5B => ("z", Cons),
        0x5C => ("r", Cons),
        0x5D => ("rh", Cons),
        0x5E => ("f", Cons),
        0x5F => ("y", Cons),
        0x3C => ("", Nukta),
        0x3E => ("a", Matra),
        0x3F | 0x40 => ("i", Matra),
        0x41 | 0x42 => ("u", Matra),
        0x43 | 0x44 => ("ri", Matra),
        0x45..=0x47 => ("e", Matra),
        0x48 => ("ai", Matra),
        0x49..=0x4B => ("o", Matra),
        0x4C => ("au", Matra),
        0x3A | 0x3B | 0x4E | 0x4F | 0x55..=0x57 | 0x62 | 0x63 => ("", Matra),
        0x4D => ("", Virama),
        0x50 => ("om", Other),
        0x64 | 0x65 => (" ", Other),
        0x66..=0x6F => (DIGITS[(off - 0x66) as usize], Other),
        // Malayalam chillu letters.
        0x7A | 0x7B => ("n", Other),
        0x7C => ("r", Other),
        0x7D | 0x7E => ("l", Other),
        0x7F => ("k", Other),
        _ => ("", Other),
    }
}

#[inline]
fn is_indic(c: char) -> bool {
    (0x0900..=0x0D7F).contains(&(c as u32))
}

fn transliterate_into(s: &str, out: &mut String) {
    let chars: Vec<char> = s.chars().collect();
    for i in 0..chars.len() {
        let c = chars[i];
        if !is_indic(c) {
            out.push(c);
            continue;
        }
        let (lat, k) = indic(c as u32 & 0x7F);
        out.push_str(lat);
        if k == K::Cons || k == K::Nukta {
            // Inherent vowel, unless followed by a vowel sign / virama / nukta, or at
            // the end of a word (schwa deletion).
            let next = chars
                .get(i + 1)
                .filter(|n| is_indic(**n))
                .map(|n| indic(*n as u32 & 0x7F).1);
            match next {
                None | Some(K::Matra) | Some(K::Virama) | Some(K::Nukta) => {}
                Some(_) => out.push('a'),
            }
        }
    }
}

/// Lowercase, transliterate Indic scripts, strip diacritics, and map punctuation to spaces.
/// `.` and apostrophes are deleted rather than spaced so "L.L.C." -> "llc", "Lord's" -> "lords".
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    let push = |c: char, out: &mut String| {
        for lc in c.to_lowercase() {
            if lc.is_alphanumeric() {
                out.push(lc);
            } else if matches!(lc, '.' | '\'' | '\u{2019}' | '`' | '\u{200b}'..='\u{200d}' | '\u{2060}' | '\u{feff}') {
            } else {
                out.push(' ');
            }
        }
    };
    if s.is_ascii() {
        for c in s.chars() {
            push(c, &mut out);
        }
    } else {
        let mut t = String::with_capacity(s.len() + 16);
        transliterate_into(s, &mut t);
        for c in t.nfkd() {
            if !is_combining_mark(c) {
                push(c, &mut out);
            }
        }
    }
    out
}

/// Strip "www." and a trailing web TLD from a raw (lowercased) word: "bethchapel.com" -> "bethchapel".
fn strip_web(w: &str) -> &str {
    let w = w.strip_prefix("www.").unwrap_or(w);
    for tld in [".co.in", ".com", ".net", ".org", ".biz", ".info", ".in", ".fr", ".co", ".us", ".io"] {
        if let Some(s) = w.strip_suffix(tld) {
            if !s.is_empty() {
                return s;
            }
        }
    }
    w
}

// ---------------------------------------------------------------------------------------
// Phonetic skeleton: consonant key robust to vowel typos, transliteration and b/v, g/j,
// c/k, p/f confusions. "Tetlecommunication" and "Telecommunication" -> same key.
// ---------------------------------------------------------------------------------------

pub fn skeleton(w: &str) -> Option<String> {
    // "-tion" is transliterated as "-shan" in Indic scripts ("international" -> "intarnyashanal").
    let w = w.replace("tion", "shn");
    let b = w.as_bytes();
    if b.len() < 3 || !b.iter().all(|c| c.is_ascii_lowercase()) {
        return None;
    }
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    if matches!(b[0], b'a' | b'e' | b'i' | b'o' | b'u') {
        out.push(b'a');
    }
    for &c in b {
        let m = match c {
            b'a' | b'e' | b'i' | b'o' | b'u' | b'y' | b'h' => continue,
            b'c' | b'k' | b'q' => b'k',
            b'g' | b'j' | b'z' => b'j',
            b'v' | b'b' | b'w' => b'b',
            b'f' | b'p' => b'p',
            b'x' => {
                out.push(b'k');
                b's'
            }
            other => other,
        };
        if out.last() != Some(&m) {
            out.push(m);
        }
    }
    if out.len() < 2 {
        return None;
    }
    Some(String::from_utf8(out).unwrap())
}

// ---------------------------------------------------------------------------------------
// Canonicalisation tables.
// ---------------------------------------------------------------------------------------

fn canon_name_word(w: &str) -> &str {
    match w {
        "pvt" | "pvte" => "private",
        "ltd" | "ltda" => "limited",
        "corp" | "corpn" => "corporation",
        "co" | "cos" => "company",
        "inc" | "incorporated" => "inc",
        "intl" => "international",
        "mfg" => "manufacturing",
        "svc" | "svcs" => "services",
        "mgmt" => "management",
        "bros" => "brothers",
        "assoc" | "assocs" => "associates",
        "ent" | "ents" => "enterprises",
        "&" => "and",
        _ => w,
    }
}

fn is_legal_or_stop(w: &str) -> bool {
    matches!(
        w,
        "private" | "limited" | "corporation" | "company" | "inc" | "llc" | "llp" | "lp" | "plc"
            | "pllc" | "sarl" | "sas" | "sasu" | "sa" | "eurl" | "sci" | "snc" | "gmbh" | "the"
            | "and" | "of" | "et" | "de" | "la" | "le" | "les" | "du" | "des" | "pty" | "dba"
    )
}

fn canon_addr_word(w: &str) -> &str {
    match w {
        "street" | "str" => "st",
        "road" => "rd",
        "avenue" | "av" | "avn" => "ave",
        "drive" | "drv" => "dr",
        "lane" => "ln",
        "boulevard" | "bd" | "boul" => "blvd",
        "court" => "ct",
        "place" => "pl",
        "trail" => "trl",
        "highway" => "hwy",
        "parkway" => "pkwy",
        "circle" => "cir",
        "suite" => "ste",
        "apartment" | "apartments" => "apt",
        "square" => "sq",
        "terrace" => "ter",
        "north" => "n",
        "south" => "s",
        "east" => "e",
        "west" => "w",
        "floor" => "fl",
        "number" | "num" => "no",
        "building" | "bldg" => "bldg",
        // US states
        "alabama" => "al", "alaska" => "ak", "arizona" => "az", "arkansas" => "ar",
        "california" => "ca", "colorado" => "co", "connecticut" => "ct", "delaware" => "de",
        "florida" => "fl", "georgia" => "ga", "hawaii" => "hi", "idaho" => "id",
        "illinois" => "il", "indiana" => "in", "iowa" => "ia", "kansas" => "ks",
        "kentucky" => "ky", "louisiana" => "la", "maine" => "me", "maryland" => "md",
        "massachusetts" => "ma", "michigan" => "mi", "minnesota" => "mn", "mississippi" => "ms",
        "missouri" => "mo", "montana" => "mt", "nebraska" => "ne", "nevada" => "nv",
        "ohio" => "oh", "oklahoma" => "ok", "oregon" => "or", "pennsylvania" => "pa",
        "tennessee" => "tn", "texas" => "tx", "utah" => "ut", "vermont" => "vt",
        "virginia" => "va", "washington" => "wa", "wisconsin" => "wi", "wyoming" => "wy",
        // Indian states / UTs
        "maharashtra" => "mh", "karnataka" => "ka", "gujarat" => "gj", "rajasthan" => "rj",
        "kerala" => "kl", "telangana" => "tg", "delhi" => "dl", "bihar" => "br",
        "odisha" | "orissa" => "od", "punjab" => "pb", "haryana" => "hr", "assam" => "as",
        "jharkhand" => "jh", "chhattisgarh" => "cg", "uttarakhand" => "uk", "goa" => "ga",
        "null" | "none" | "nan" | "na" => "",
        _ => w,
    }
}

fn canon_addr_bigram(a: &str, b: &str) -> Option<&'static str> {
    Some(match (a, b) {
        ("new", "hampshire") => "nh",
        ("new", "jersey") => "nj",
        ("new", "mexico") => "nm",
        ("new", "york") => "ny",
        ("north", "carolina") => "nc",
        ("north", "dakota") => "nd",
        ("south", "carolina") => "sc",
        ("south", "dakota") => "sd",
        ("west", "virginia") => "wv",
        ("rhode", "island") => "ri",
        ("west", "bengal") => "wb",
        ("tamil", "nadu") => "tn",
        ("uttar", "pradesh") => "up",
        ("madhya", "pradesh") => "mp",
        ("andhra", "pradesh") => "ap",
        ("himachal", "pradesh") => "hp",
        ("arunachal", "pradesh") => "ar",
        _ => return None,
    })
}

/// Strip leading zeros from number-led tokens ("01018" -> "1018", "05th" -> "5th").
fn canon_number(w: &str) -> &str {
    if w.len() < 2 || !w.starts_with('0') {
        return w;
    }
    let t = w.trim_start_matches('0');
    match t.as_bytes().first() {
        None => "0",
        Some(c) if c.is_ascii_digit() => t,
        _ => w,
    }
}

// ---------------------------------------------------------------------------------------
// Tokenisation into hashed blocking keys.
// ---------------------------------------------------------------------------------------

pub fn country_key(country: &str) -> u64 {
    let mut h = FxHasher::default();
    h.write(country.trim().to_lowercase().as_bytes());
    h.finish()
}

#[inline]
fn tok(country: u64, ch: u64, tag: u8, s: &str) -> u64 {
    let mut h = FxHasher::default();
    h.write_u64(country);
    h.write_u8(tag);
    h.write(s.as_bytes());
    // FxHash's low bits are weak; fold high bits down before masking in the channel.
    let v = h.finish();
    let v = v ^ (v >> 29) ^ (v >> 47);
    (v & !7) | ch
}

#[inline]
pub fn channel(t: u64) -> usize {
    (t & 7) as usize
}

/// Normalised name words (with legal-form canonicalisation).
pub fn name_words(name: &str) -> Vec<String> {
    let lower = name.to_lowercase();
    let mut pre = String::with_capacity(lower.len());
    for w in lower.split_whitespace() {
        pre.push_str(strip_web(w));
        pre.push(' ');
    }
    normalize(&pre)
        .split_whitespace()
        .map(|w| canon_name_word(w).to_string())
        .collect()
}

pub fn addr_words(addr: &str) -> Vec<String> {
    let norm = normalize(addr);
    let ws: Vec<&str> = norm.split_whitespace().collect();
    let mut out = Vec::with_capacity(ws.len());
    let mut i = 0;
    while i < ws.len() {
        if i + 1 < ws.len() {
            if let Some(code) = canon_addr_bigram(ws[i], ws[i + 1]) {
                out.push(code.to_string());
                i += 2;
                continue;
            }
        }
        let w = canon_number(canon_addr_word(ws[i]));
        if !w.is_empty() {
            out.push(w.to_string());
        }
        i += 1;
    }
    out
}

/// Append all hashed blocking tokens of one record to `out` (sorted, deduplicated).
pub fn record_tokens(name: &str, addr: &str, country: &str, out: &mut Vec<u64>) {
    let c = country_key(country);
    let start = out.len();

    let nw = name_words(name);
    let mut core: Vec<&str> = Vec::new();
    let mut tri: Vec<String> = Vec::new();
    for w in &nw {
        out.push(tok(c, CH_NAME, b'w', w));
        if let Some(s) = skeleton(w) {
            out.push(tok(c, CH_NAME_SKEL, b's', &s));
        }
        // Prefix/suffix keys catch single typos in long words ("felloship" ~ "fellowship").
        if w.len() >= 6 && w.is_ascii() {
            out.push(tok(c, CH_NAME_SKEL, b'p', &w[..4]));
            out.push(tok(c, CH_NAME_SKEL, b'x', &w[w.len() - 4..]));
        }
        if !is_legal_or_stop(w) {
            trigrams(w, &mut tri);
            for t in &tri {
                out.push(tok(c, CH_TRIG, b't', t));
            }
            core.push(w);
        }
    }
    // Joined keys: robust to word splits/joins ("beth chapel" == "bethchapel") and,
    // sorted, to word-order transpositions.
    if !core.is_empty() {
        out.push(tok(c, CH_NAME_JOIN, b'j', &core.concat()));
        if core.len() > 1 {
            let mut sorted = core.clone();
            sorted.sort_unstable();
            out.push(tok(c, CH_NAME_JOIN, b'j', &sorted.concat()));
        }
    }

    for w in addr_words(addr) {
        out.push(tok(c, CH_ADDR, b'a', &w));
        if w.len() >= 4 {
            if let Some(s) = skeleton(&w) {
                out.push(tok(c, CH_ADDR_SKEL, b'k', &s));
            }
        }
    }

    out[start..].sort_unstable();
    let mut w = start;
    for r in start..out.len() {
        if r == start || out[r] != out[w - 1] {
            out[w] = out[r];
            w += 1;
        }
    }
    out.truncate(w);
}

/// Character trigrams (`#`-padded) of one normalized word, e.g. "grand" ->
/// ["#gr", "gra", "ran", "and", "nd#"]. ASCII alphanumeric words only.
pub fn trigrams(w: &str, out: &mut Vec<String>) {
    if w.len() < 4 || !w.is_ascii() || !w.bytes().all(|c| c.is_ascii_alphanumeric()) {
        return;
    }
    let p = format!("#{w}#");
    let b = p.as_bytes();
    out.clear();
    for i in 0..b.len() - 2 {
        out.push(String::from_utf8_lossy(&b[i..i + 3]).into_owned());
    }
}
/// Stable 64-bit FNV-1a used for the deterministic train/validation split.
/// Mirror this in Python for Part 2: `val = fnv1a64(id) % 10 == 0`.
pub fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translit_devanagari() {
        assert_eq!(normalize("बालाजी"), "balaji");
        assert_eq!(normalize("अरिहंत"), "arihant");
    }

    #[test]
    fn skeleton_matches_typo_and_script() {
        assert_eq!(skeleton("tetlecommunication"), skeleton("telecommunication"));
        assert_eq!(skeleton(&normalize("प्राइवेट")), skeleton("private"));
        assert_eq!(skeleton(&normalize("इंजीनियरिंग")), skeleton("engineering"));
        assert_eq!(skeleton(&normalize("டெக்னாலஜீஸ்")), skeleton("technologies"));
        assert_eq!(normalize("ಇಂಟರ್\u{200c}ನ್ಯಾಷನಲ್").split_whitespace().count(), 1);
        assert_eq!(skeleton(&normalize("ಇಂಟರ್\u{200c}ನ್ಯಾಷನಲ್")), skeleton("international"));
    }

    #[test]
    fn trigrams_catch_compounds_and_typos() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        trigrams("infrastructureprivate", &mut a);
        trigrams("infrastructure", &mut b);
        // every interior trigram of the short word appears in the compound
        assert!(b.iter().filter(|t| !t.starts_with('#') && !t.ends_with('#')).all(|t| a.contains(t)));
        let mut c = Vec::new();
        let mut d = Vec::new();
        trigrams("grand", &mut c);
        trigrams("6rand", &mut d);
        assert!(c.iter().any(|t| d.contains(t)));
        // too short / non-alphanumeric words give nothing
        let mut e = Vec::new();
        trigrams("co", &mut e);
        assert!(e.is_empty());
    }

    #[test]
    fn accents_and_legal_forms() {
        assert_eq!(name_words("Anitra Vásquez Money L.L.C."), vec!["anitra", "vasquez", "money", "llc"]);
        assert_eq!(name_words("bethchapel.com"), vec!["bethchapel"]);
        assert_eq!(addr_words("05Th Floor, 0046"), vec!["5th", "fl", "46"]);
        assert_eq!(addr_words("01018 KENWOOD ST, HAMMOND, IN"), vec!["1018", "kenwood", "st", "hammond", "in"]);
        assert_eq!(addr_words("1018 Kenwood Street, Glenville, North Carolina"), vec!["1018", "kenwood", "st", "glenville", "nc"]);
    }
}
