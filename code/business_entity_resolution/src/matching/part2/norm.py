"""Country-agnostic text normalisation and tokenisation for Part 2 features.

Mirrors the ideas in the Part 1 Rust pipeline (code/business_entity_resolution/src/
business_candy_crush/src/normalize.rs) where it matters for features: Indic -> Latin
transliteration, accent stripping, legal-form and street-type canonicalisation, leading
zero removal. Everything is pure-Python + stdlib and deterministic across runs.
"""

import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------------------
# Indic -> Latin transliteration (ISCII-derived layout, U+0900..U+0D7F).
# Table keyed by codepoint & 0x7F: (latin, kind), kind in C V M N X
# ---------------------------------------------------------------------------
_VOWELS = "aeiou"

# kind: 0=consonant 1=vowel 2=matra 3=virama 4=nukta 5=other
_INDIC = {
    0x01: ("n", 5), 0x02: ("n", 5), 0x03: ("h", 5),
    0x05: ("a", 1), 0x06: ("a", 1), 0x07: ("i", 1), 0x08: ("i", 1),
    0x09: ("u", 1), 0x0A: ("u", 1), 0x0B: ("ri", 1), 0x0C: ("li", 1),
    0x0D: ("e", 1), 0x0E: ("e", 1), 0x0F: ("e", 1), 0x10: ("ai", 1),
    0x11: ("o", 1), 0x12: ("o", 1), 0x13: ("o", 1), 0x14: ("au", 1),
    0x15: ("k", 0), 0x16: ("kh", 0), 0x17: ("g", 0), 0x18: ("gh", 0),
    0x19: ("n", 0), 0x1A: ("ch", 0), 0x1B: ("chh", 0), 0x1C: ("j", 0),
    0x1D: ("jh", 0), 0x1E: ("n", 0), 0x1F: ("t", 0), 0x20: ("th", 0),
    0x21: ("d", 0), 0x22: ("dh", 0), 0x23: ("n", 0), 0x24: ("t", 0),
    0x25: ("th", 0), 0x26: ("d", 0), 0x27: ("dh", 0), 0x28: ("n", 0),
    0x29: ("n", 0), 0x2A: ("p", 0), 0x2B: ("ph", 0), 0x2C: ("b", 0),
    0x2D: ("bh", 0), 0x2E: ("m", 0), 0x2F: ("y", 0), 0x30: ("r", 0),
    0x31: ("r", 0), 0x32: ("l", 0), 0x33: ("l", 0), 0x34: ("l", 0),
    0x35: ("v", 0), 0x36: ("sh", 0), 0x37: ("sh", 0), 0x38: ("s", 0),
    0x39: ("h", 0),
    0x58: ("q", 0), 0x59: ("kh", 0), 0x5A: ("g", 0), 0x5B: ("z", 0),
    0x5C: ("r", 0), 0x5D: ("rh", 0), 0x5E: ("f", 0), 0x5F: ("y", 0),
    0x3C: ("", 4),
    0x3E: ("a", 2), 0x3F: ("i", 2), 0x40: ("i", 2), 0x41: ("u", 2),
    0x42: ("u", 2), 0x43: ("ri", 2), 0x44: ("ri", 2), 0x45: ("e", 2),
    0x46: ("e", 2), 0x47: ("e", 2), 0x48: ("ai", 2), 0x49: ("o", 2),
    0x4A: ("o", 2), 0x4B: ("o", 2), 0x4C: ("au", 2),
    0x3A: ("", 2), 0x3B: ("", 2), 0x4E: ("", 2), 0x4F: ("", 2),
    0x55: ("", 2), 0x56: ("", 2), 0x57: ("", 2), 0x62: ("", 2), 0x63: ("", 2),
    0x4D: ("", 3),
    0x50: ("om", 5), 0x64: (" ", 5), 0x65: (" ", 5),
    0x7A: ("n", 5), 0x7B: ("n", 5), 0x7C: ("r", 5), 0x7D: ("l", 5), 0x7E: ("l", 5),
    0x7F: ("k", 5),
}
for _d in range(10):
    _INDIC[0x66 + _d] = (str(_d), 5)


def _translit(s: str) -> str:
    """Indic -> Latin, mirroring the Rust `transliterate_into` (inherent vowel +
    schwa deletion rules)."""
    out = []
    n = len(s)
    for i, c in enumerate(s):
        o = ord(c)
        if not (0x0900 <= o <= 0x0D7F):
            out.append(c)
            continue
        lat, kind = _INDIC.get(o & 0x7F, ("", 5))
        out.append(lat)
        if kind in (0, 4):  # consonant / nukta: inherent 'a' unless followed by matra/virama/nukta or word-end (schwa deletion)
            nxt = s[i + 1] if i + 1 < n else None
            nk = None
            if nxt is not None and 0x0900 <= ord(nxt) <= 0x0D7F:
                nk = _INDIC.get(ord(nxt) & 0x7F, ("", 5))[1]
            if nk is not None and nk not in (2, 3, 4):
                out.append("a")
    return "".join(out)


_DROP = {ord(c): None for c in "\u2019\u200b\u200c\u200d\u2060\ufeff"}
# punctuation-ish chars map to space; dot and apostrophes are deleted (done separately)
_SPACE_TRANS = str.maketrans({c: " " for c in "!\"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\r"})
_SPACE_TRANS.update(_DROP)
_SPACE_TRANS.update({ord("."): None, ord("'"): None})


@lru_cache(maxsize=1 << 20)
def normalize(s: str) -> str:
    """Lowercase, transliterate Indic scripts, strip accents and punctuation.
    Output contains only [a-z0-9 ] (other scripts' letters are kept)."""
    if not s:
        return ""
    if s.isascii():
        t = s.lower().translate(_SPACE_TRANS)
    else:
        if any(0x0900 <= ord(c) <= 0x0D7F for c in s):
            s = _translit(s)
        t = "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                    if not unicodedata.combining(c))
        t = t.translate(_SPACE_TRANS)
    return " ".join(t.split())


# ---------------------------------------------------------------------------
# Canonicalisation tables (shared with the Rust pipeline's spirit).
# ---------------------------------------------------------------------------

_LEGAL = {"pvt": "private", "pvte": "private", "ltd": "limited", "ltda": "limited",
          "corp": "corporation", "corpn": "corporation", "co": "company", "cos": "company",
          "inc": "inc", "incorporated": "inc", "intl": "international",
          "mfg": "manufacturing", "svc": "services", "svcs": "services",
          "mgmt": "management", "bros": "brothers", "assoc": "associates",
          "assocs": "associates", "ent": "enterprises", "ents": "enterprises",
          "llp": "llp", "llc": "llc", "plc": "plc", "gmbh": "gmbh"}

_STOP = {"private", "limited", "corporation", "company", "inc", "llc", "llp", "lp", "plc",
         "pllc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "gmbh", "the", "and",
         "of", "et", "de", "la", "le", "les", "du", "des", "pty", "dba", "trade", "name"}

_ADDR = {"street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
         "drive": "dr", "drv": "dr", "lane": "ln", "boulevard": "blvd", "bd": "blvd",
         "boul": "blvd", "court": "ct", "place": "pl", "trail": "trl", "highway": "hwy",
         "parkway": "pkwy", "circle": "cir", "suite": "ste", "apartment": "apt",
         "apartments": "apt", "square": "sq", "terrace": "ter", "north": "n", "south": "s",
         "east": "e", "west": "w", "floor": "fl", "number": "no", "num": "no",
         "building": "bldg", "bldg": "bldg", "near": "", "opp": "opp", "oppo": "opp",
         "null": "", "none": "", "nan": "", "na": ""}

_ADDR_BIGRAM = {("new", "hampshire"): "nh", ("new", "jersey"): "nj", ("new", "mexico"): "nm",
                ("new", "york"): "ny", ("north", "carolina"): "nc", ("north", "dakota"): "nd",
                ("south", "carolina"): "sc", ("south", "dakota"): "sd",
                ("west", "virginia"): "wv", ("rhode", "island"): "ri",
                ("west", "bengal"): "wb", ("tamil", "nadu"): "tn", ("uttar", "pradesh"): "up",
                ("madhya", "pradesh"): "mp", ("andhra", "pradesh"): "ap",
                ("himachal", "pradesh"): "hp", ("arunachal", "pradesh"): "ar"}

_STATES = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
           "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
           "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
           "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
           "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
           "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
           "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
           "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
           "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
           "wyoming": "wy", "maharashtra": "mh", "karnataka": "ka", "gujarat": "gj",
           "rajasthan": "rj", "kerala": "kl", "telangana": "tg", "telunga": "tg",
           "delhi": "dl", "bihar": "br", "odisha": "od", "orissa": "od", "punjab": "pb",
           "haryana": "hr", "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg",
           "uttarakhand": "uk", "goa": "ga"}

_TLDS = (".co.in", ".com", ".net", ".org", ".biz", ".info", ".in", ".fr", ".co", ".us", ".io", ".www")


def _strip_web(w: str) -> str:
    if w.startswith("www."):
        w = w[4:]
    for t in _TLDS:
        if w.endswith(t):
            s = w[: -len(t)]
            if s:
                return s
    return w


def _canon_number(w: str) -> str:
    if len(w) >= 2 and w[0] == "0":
        t = w.lstrip("0")
        if t and t[0].isdigit():
            return t
        if not t:
            return "0"
    return w


def name_tokens(name: str) -> list:
    """Canonicalised normalised name words (legal forms expanded)."""
    parts = []
    for w in name.lower().split():
        for p in normalize(_strip_web(w)).split():
            parts.append(_LEGAL.get(p, p))
    return [p for p in parts if p]


def name_core(name: str) -> list:
    """Name words without legal suffixes / stop words (for joined keys)."""
    return [w for w in name_tokens(name) if w not in _STOP]


def addr_tokens(addr: str) -> list:
    """Normalised, canonicalised address words (state bigrams folded, zeros stripped)."""
    ws = normalize(addr).split()
    out = []
    i = 0
    while i < len(ws):
        if i + 1 < len(ws):
            big = _ADDR_BIGRAM.get((ws[i], ws[i + 1]))
            if big:
                out.append(big)
                i += 2
                continue
        w = ws[i]
        c = _ADDR.get(w, w)
        if c == "":
            i += 1
            continue
        c = _canon_number(c)
        c = _STATES.get(c, c)
        if c:
            out.append(c)
        i += 1
    return out


# ---------------------------------------------------------------------------
# Hashing helpers. blake2b is stable across processes (unlike Python's hash()).
# ---------------------------------------------------------------------------

import hashlib

CAP_NAME = 14   # per-token-set capacity for exact pairwise intersections
CAP_ADDR = 20


def hash64(w: str) -> int:
    d = hashlib.blake2b(w.encode("utf-8", "ignore"), digest_size=4).digest()
    return int.from_bytes(d, "little") | 1  # 0 reserved as pad value


def hash_words(tokens, cap: int = CAP_NAME):
    """(sorted u32 hashes capped at `cap`, total distinct count)."""
    seen = {hash64(t) for t in tokens}
    vals = sorted(seen)[:cap]
    return vals, len(seen)
