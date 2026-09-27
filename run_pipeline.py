#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026 — Pipeline V26 (Ultimate Champion Strategy)
================================================================================
Precision & Recall Leveled-Up Strategy:
  - Inverted Municipal Syntax Normalization ("Denison, Borough Of" -> "borough of denison")
  - Address Prefix Normalization (khasra, door, plot, house, flat, ward, sector)
  - 3-Number House/Unit Compound Indexing (cand_hnums[:3])
  - Adaptive Dual-Window Post-Processing:
      * max_conf >= 0.93: delta=0.07, max_matches=6 (High Recall for Confident Clusters)
      * 0.90 <= max_conf < 0.93: delta=0.05, max_matches=5 (High Precision Guard)
      * max_conf < 0.90: empty (Singleton Protection)
  - Generates both matching_results.tsv and candidate_pairs.tsv directly!
================================================================================
"""

import os
import sys
import gc
import re
import time
import shutil
import unicodedata
from collections import defaultdict
import subprocess

print("=" * 80)
print(" Amazon ML Challenge 2026 — Fine-Tuned Pipeline (V26 Champion)")
print("=" * 80, flush=True)

# Install rapidfuzz and polars if needed
for pkg in ["rapidfuzz", "polars"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"[*] Installing {pkg}...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import polars as pl
from rapidfuzz import fuzz

# Hardware check
try:
    import torch
    if torch.cuda.is_available():
        print(f"[+] GPU Detected: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("[+] CPU Multi-Core Engine Active", flush=True)
except Exception:
    pass

# Locate dataset
test_dir = None
search_candidates = [
    "student_resource/dataset/test",
    "dataset/test",
    "/kaggle/input",
]

for cand in search_candidates:
    if os.path.exists(os.path.join(cand, "test_source1.tsv")):
        test_dir = cand
        break

if test_dir is None:
    for root, _, files in os.walk("/kaggle/input" if os.path.exists("/kaggle/input") else "."):
        if "test_source1.tsv" in files:
            test_dir = root
            break

if test_dir is None:
    for root, _, files in os.walk("/kaggle/input" if os.path.exists("/kaggle/input") else "."):
        if "student_resource.tar.gz" in files:
            tar_path = os.path.join(root, "student_resource.tar.gz")
            print(f"[*] Extracting {tar_path}...", flush=True)
            import tarfile
            extract_dir = "/kaggle/working/extracted_dataset" if os.path.exists("/kaggle/working") else "extracted_dataset"
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            for eroot, _, efiles in os.walk(extract_dir):
                if "test_source1.tsv" in efiles:
                    test_dir = eroot
                    break
            break

print(f"[+] Test Directory: {test_dir}", flush=True)

s1_path = os.path.join(test_dir, "test_source1.tsv")
s2_path = os.path.join(test_dir, "test_source2.tsv")
s3_path = os.path.join(test_dir, "test_source3.tsv")

_COUNTRY_MAP = {
    "usa": "us", "united states": "us", "uk": "gb",
    "united kingdom": "gb", "great britain": "gb",
    "deutschland": "de", "germany": "de", "france": "fr",
    "italia": "it", "italy": "it", "espana": "es", "spain": "es",
    "india": "in", "china": "cn", "japan": "jp", "canada": "ca",
    "australia": "au", "brazil": "br", "brasil": "br", "mexico": "mx",
}

LEGAL_SUFFIXES = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "llc", "gmbh", "sa", "sarl", "sas", "sasu", "eurl", "sci", "snc", "scop", "ei",
    "sl", "bv", "co", "company", "pvt", "private", "plc", "srl", "pty",
    "holdings", "group", "mr", "mrs", "dr", "dds", "services", "service",
    "center", "centre", "solutions", "enterprises", "enterprise", "partners",
    "consulting", "associates", "llp", "public", "pllc", "pc", "pa",
    "traders", "agency", "agencies", "stores",
}

_ADDR_COMMON = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "sq": "square", "ste": "suite", "apt": "apartment", "dept": "department",
    "fl": "floor", "bldg": "building", "hwy": "highway", "pkwy": "parkway",
    "str": "strasse", "pl": "place", "terr": "terrace",
    "cir": "circle", "trl": "trail", "twp": "township",
    "bd": "boulevard", "bvd": "boulevard", "ch": "chemin",
    "imp": "impasse", "rte": "route", "crs": "cours", "al": "allee",
    "fg": "faubourg", "pass": "passage", "qu": "quai",
    # Ordinals
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth",
}

_ADDR_US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming",
}

_ADDR_IN_STATES = {
    "mh": "maharashtra", "dl": "delhi", "ka": "karnataka", "tn": "tamil nadu",
    "up": "uttar pradesh", "wb": "west bengal", "rj": "rajasthan", "gj": "gujarat",
    "ts": "telangana", "tg": "telangana", "ap": "andhra pradesh", "kl": "kerala",
    "mp": "madhya pradesh", "hr": "haryana", "pb": "punjab", "br": "bihar",
    "od": "odisha", "or": "odisha", "cg": "chhattisgarh", "ch": "chandigarh",
    "jh": "jharkhand", "as": "assam", "uk": "uttarakhand", "ut": "uttarakhand",
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai",
}

_INDIC_STATE_MAP = {
    "दिल्ली": "delhi", "महाराष्ट्र": "maharashtra", "उत्तर": "uttar", "प्रदेश": "pradesh",
    "गुजरात": "gujarat", "ಕರ್ನಾಟಕ": "karnataka", "தமிழ்நாடு": "tamil nadu",
    "পশ্চিমবঙ্গ": "west bengal", "తెలంగాణ": "telangana", "കേരളം": "kerala",
    "हरियाणा": "haryana", "राजस्थान": "rajasthan", "बिहार": "bihar",
    "मध्य": "madhya", "ਪੰਜਾਬ": "punjab", "ଓଡ଼ିଶା": "odisha",
    "ఆంధ్రప్రదేశ్": "andhra pradesh", "ಆಂಧ್ರಪ್ರದೇಶ": "andhra pradesh",
}

GENERIC_STREET_WORDS = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "court", "square",
    "suite", "apartment", "department", "floor", "building", "highway", "parkway",
    "strasse", "place", "terrace", "near", "opposite", "behind", "block", "sector",
    "post", "unit", "town", "city", "county", "state", "main", "cross",
    # French & generic additions
    "rue", "chemin", "impasse", "route", "cours", "allee", "faubourg", "quai",
    "cedex", "bis", "ter", "quater", "circle", "trail", "township", "loop", "way", "path",
}

COMMON_PREFIX_WORDS = {
    "the", "st", "saint", "hotel", "cafe", "restaurant", "club", "store", "shop", "bar",
    "national", "international", "global", "royal", "first", "american", "indian"
}

HONORIFIC_PREFIXES = {
    "m/s", "shri", "smt", "sri", "mr", "mrs", "dr", "dds", "ms", "the"
}


def norm_country(c):
    if not c or not isinstance(c, str):
        return ""
    return _COUNTRY_MAP.get(c.strip().lower(), c.strip().lower())


def strip_accents(text):
    if not text:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def ocr_normalize(s):
    if not s:
        return ""
    s_norm = re.sub(r'(?<=[a-z])0(?=[a-z])', 'o', s)
    s_norm = re.sub(r'(?<=[a-z])1(?=[a-z])', 'l', s_norm)
    s_norm = re.sub(r'(?<=[a-z])5(?=[a-z])', 's', s_norm)
    return s_norm


def clean_name(s):
    if not s or not isinstance(s, str):
        return ""
    s = strip_accents(s.lower())
    if " dba " in s:
        s = s.split(" dba ")[-1]
    elif " aka " in s:
        s = s.split(" aka ")[-1]
    s = re.sub(r'\(id:?\s*\d+\)', '', s)
    # Strip parenthetical country tags
    s = re.sub(r'\((india|france|usa|us|uk|canada|germany|deutschland|italy|spain)\)', '', s)
    s = re.sub(r"['’]s\b", "", s)
    # Inverted municipal name syntax (e.g. "Denison, Borough Of" -> "borough of denison")
    s = re.sub(r'^([^,]+),\s*(borough|city|town|village|township|department|division|county|state|office|board)\s+of\b', r'\2 of \1', s)
    # Strip leading noise symbols
    s = re.sub(r'^[<>{}\[\]@#*~|\\/\-\s]+', '', s)
    # Strip trailing phone numbers / long IDs (e.g. - 6165510308)
    s = re.sub(r'[-–—#]?\s*\b\d{7,12}\b', '', s)
    # Strip store / branch codes (e.g. #68886)
    s = re.sub(r'#\s*\d+\b', '', s)
    # Strip URLs and domains
    s = re.sub(r'\|\s*www\.[^\s]+', '', s)
    s = re.sub(r'\bwww\.', '', s)
    s = re.sub(r'\.(com|org|net|co\.in|in|io|biz|fr|co|info|edu|gov)\b', '', s)
    # Handle m/s specifically
    s = re.sub(r'\bm/s\b', '', s)
    s = re.sub(r"[^\w\s]", " ", s)
    words = s.split()
    # Strip leading honorifics / noise words
    while words and words[0] in HONORIFIC_PREFIXES:
        words.pop(0)
    # Strip leading legal suffixes that were moved to the front
    while words and words[0] in {"private", "pvt", "public", "limited", "ltd", "inc", "corp", "llc", "sarl", "sas", "sasu", "eurl", "sci"}:
        words.pop(0)
    return " ".join(words)


def clean_addr(s, country=""):
    if not s or not isinstance(s, str):
        return ""
    s = strip_accents(s.lower())
    # Clean null placeholders
    s = re.sub(r'<null>', ' ', s)
    s = re.sub(r'\bnull\b', ' ', s)
    s = re.sub(r'\bn/a\b', ' ', s)
    # Normalize address prefixes
    s = re.sub(r'\b(kh|khasra)\.?\s*no\.?', 'khasra', s)
    s = re.sub(r'\b(d|door)\.?\s*no\.?', 'door', s)
    s = re.sub(r'\b(pl|plot)\.?\s*no\.?', 'plot', s)
    s = re.sub(r'\b(h|house)\.?\s*no\.?', 'house', s)
    s = re.sub(r'\b(fl|flat)\.?\s*no\.?', 'flat', s)
    # Map Indic state names
    for k, v in _INDIC_STATE_MAP.items():
        if k in s:
            s = s.replace(k, v)
    s = re.sub(r"[^\w\s/]", " ", s)
    words = []
    for w in s.split():
        if w in _ADDR_COMMON:
            words.append(_ADDR_COMMON[w])
        elif country == "us" and w in _ADDR_US_STATES:
            words.append(_ADDR_US_STATES[w])
        elif country == "in" and w in _ADDR_IN_STATES:
            words.append(_ADDR_IN_STATES[w])
        else:
            words.append(w)
    res = " ".join(words)
    return re.sub(r"/+", " ", res)


def core_name(cn):
    if not cn:
        return ""
    words = [w for w in cn.split() if w not in LEGAL_SUFFIXES]
    return " ".join(words)


def name_no_space(cn):
    if not cn:
        return ""
    s = cn.replace(" ", "")
    for suf in ["com", "org", "net", "inc", "llc", "ltd", "corp", "sas", "sarl"]:
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s


def extract_addr_features(raw_addr, cleaned_addr, country):
    if not cleaned_addr:
        return "", "", set(), [], []

    all_num_tokens = [x.lstrip("0") for x in re.findall(r"\b\d+\b", cleaned_addr) if x.lstrip("0")]
    nums_set = set(all_num_tokens)

    postal_code = ""
    target_str = raw_addr or cleaned_addr
    if country in ("us", "de", "fr", "it", "es", "mx"):
        m = re.search(r"\b\d{5}\b", target_str)
        if m:
            postal_code = m.group(0)
    elif country == "in":
        m = re.search(r"\b\d{6}\b", target_str)
        if m:
            postal_code = m.group(0)
    elif country == "ca":
        m = re.search(r"\b[a-z]\d[a-z]\s*\d[a-z]\d\b", target_str.lower())
        if m:
            postal_code = m.group(0).replace(" ", "")
    elif country == "gb":
        m = re.search(r"\b[a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2}\b", target_str.lower())
        if m:
            postal_code = m.group(0).replace(" ", "")
    elif country == "au":
        m = re.search(r"\b\d{4}\b", target_str)
        if m:
            postal_code = m.group(0)
    elif country == "br":
        m = re.search(r"\b\d{5}-?\d{3}\b", target_str)
        if m:
            postal_code = m.group(0).replace("-", "")

    cand_hnums = [n for n in all_num_tokens if n != postal_code and len(n) <= 6]
    house_num = cand_hnums[0] if cand_hnums else (all_num_tokens[0] if all_num_tokens else "")

    words = [w for w in cleaned_addr.split() if not w.isdigit() and len(w) >= 3 and w not in GENERIC_STREET_WORDS]
    return postal_code, house_num, nums_set, words, cand_hnums[:3]


def extract_blocking_keys(country, name, raw_addr, c_addr=None):
    c = norm_country(country)
    cn = clean_name(name)
    ca = clean_addr(raw_addr, c) if c_addr is None else c_addr
    cr = core_name(cn)

    keys = []
    if cn:
        keys.append(("N_EXACT", c, cn))
        if cr and cr != cn:
            keys.append(("N_CORE", c, cr))
        words = cn.split()
        if len(words) >= 2:
            keys.append(("N_FIRST2", c, f"{words[0]} {words[1]}"))
            keys.append(("N_SORTED", c, " ".join(sorted(words))))
        nosp = name_no_space(cn)
        if len(nosp) >= 5:
            keys.append(("N_NOSP", c, nosp))

    if ca:
        keys.append(("A_EXACT", c, ca))
        postal_code, house_num, nums_set, street_words, multi_hnums = extract_addr_features(raw_addr, ca, c)
        for h in multi_hnums:
            if postal_code and h:
                keys.append(("ZIP_NUM", c, f"{postal_code}_{h}"))
            if h and street_words:
                keys.append(("NUM_STREET", c, f"{h}_{street_words[0]}"))
            if h and cn:
                w0 = cn.split()[0]
                if len(w0) >= 3:
                    keys.append(("NA_NUM", c, f"{h}_{w0}"))

    return keys


def is_acronym(short_name, long_name):
    if not short_name or not long_name:
        return False
    short_name = short_name.replace(" ", "").lower()
    if not (2 <= len(short_name) <= 6):
        return False
    words = [w for w in long_name.lower().split() if w not in LEGAL_SUFFIXES]
    if len(words) < len(short_name):
        return False
    acr = "".join(w[0] for w in words if w)
    if short_name == acr[:len(short_name)]:
        return True
    stop = {"of", "the", "and", "for", "in", "on", "at", "to", "a", "an"}
    words_nostop = [w for w in words if w not in stop]
    acr_nostop = "".join(w[0] for w in words_nostop if w)
    return short_name == acr_nostop


def score_match(s1_cn, s1_ca, s1_cr, s1_nums, s1_hnum, s1_swords, s1_raw_ad,
                t_cn, t_ca, t_cr, t_nums, t_hnum, t_swords, t_raw_ad,
                country_match, s1_c, raw_t_name):
    if not country_match:
        return None, 0.0

    if not s1_cn or not t_cn:
        return None, 0.0

    has_addr = bool(s1_ca and t_ca)
    name_exact = (s1_cn == t_cn)
    s1_words = set(s1_cn.split())
    t_words = set(t_cn.split())
    common_words = s1_words & t_words
    name_ratio = 100 if name_exact else fuzz.ratio(s1_cn, t_cn)

    if not has_addr:
        if name_exact and len(s1_cr.split()) >= 3:
            return "R_NAME_EXACT_NO_ADDR", 0.89
        return None, 0.0

    num_overlap = bool(s1_nums and t_nums and (s1_nums & t_nums))
    addr_tsr = fuzz.token_set_ratio(s1_ca, t_ca)

    if addr_tsr < 55 and not (name_exact and num_overlap):
        if not (name_exact and len(s1_words) >= 3 and addr_tsr >= 45):
            return None, 0.0

    # Compound Slash Number Conflict Check (e.g. 46/1106 vs 46/1115, 37/4 vs 37/25)
    s1_target = s1_raw_ad or s1_ca
    t_target = t_raw_ad or t_ca
    s1_slash = re.findall(r"\b(\d+)/(\d+)\b", s1_target)
    t_slash = re.findall(r"\b(\d+)/(\d+)\b", t_target)
    if s1_slash and t_slash:
        if s1_slash[0][0] == t_slash[0][0] and s1_slash[0][1] != t_slash[0][1]:
            if name_ratio < 95:
                return None, 0.0

    # House & Suite Number Conflict Check with Digit-Typo Tolerance
    digit_typo_matched = False
    if s1_hnum and t_hnum and s1_hnum != t_hnum:
        if s1_hnum not in t_nums and t_hnum not in s1_nums:
            is_prefix_typo = (abs(len(s1_hnum) - len(t_hnum)) <= 1 and
                              (s1_hnum.startswith(t_hnum) or t_hnum.startswith(s1_hnum)))
            name_strong = (name_ratio >= 80 or
                           fuzz.token_set_ratio(s1_cn, t_cn) >= 80 or
                           (s1_cr and t_cr and (s1_cr.startswith(t_cr) or t_cr.startswith(s1_cr))))
            if is_prefix_typo and name_strong and len(s1_words) >= 2:
                digit_typo_matched = True
            else:
                return None, 0.0

    effective_num_overlap = num_overlap or digit_typo_matched

    # Tenant conflict check (exempting generic prefixes)
    if s1_cr and t_cr:
        w1_1 = s1_cr.split()[0]
        w1_2 = t_cr.split()[0]
        if w1_1 not in COMMON_PREFIX_WORDS and w1_2 not in COMMON_PREFIX_WORDS:
            if len(w1_1) >= 4 and len(w1_2) >= 4 and fuzz.ratio(w1_1, w1_2) < 60:
                if name_ratio < 85:
                    return None, 0.0

    if name_ratio < 45 and not common_words:
        s1_ocr = ocr_normalize(s1_cn)
        t_ocr = ocr_normalize(t_cn)
        if s1_ocr == t_ocr:
            name_ratio = 95
        else:
            return None, 0.0

    # Generic short name protection
    if s1_cr and len(s1_cr.split()) <= 1 and len(s1_cr) <= 8:
        if not effective_num_overlap or addr_tsr < 75:
            return None, 0.0

    # Rules R1-R13
    if name_ratio >= 95 and effective_num_overlap and addr_tsr >= 65:
        return "R1_NAME95_NUM_ADDR65", 0.98

    if name_exact and addr_tsr >= 78:
        return "R2_EXACT_ADDR78", 0.96

    if name_ratio >= 90 and effective_num_overlap and addr_tsr >= 62:
        return "R3_NAME90_NUM_ADDR62", 0.95

    # Core name exact + number overlap -> 0.96
    if s1_cr and t_cr:
        core_exact = (s1_cr == t_cr and len(s1_cr.split()) >= 2)
        core_ratio = fuzz.ratio(s1_cr, t_cr) if not core_exact else 100
        if core_exact and effective_num_overlap and addr_tsr >= 60:
            return "R5_CORE_EXACT_NUM", 0.96
        if core_exact and addr_tsr >= 78:
            return "R5_CORE_EXACT_ADDR78", 0.93
        if core_ratio >= 95 and effective_num_overlap and addr_tsr >= 65:
            return "R5_CORE95_NUM", 0.93

    if name_ratio >= 88 and addr_tsr >= 82 and (effective_num_overlap or addr_tsr >= 86):
        return "R4_NAME88_ADDR82", 0.94
    if name_ratio >= 92 and addr_tsr >= 78:
        return "R4_NAME92_ADDR78", 0.93

    s1_sorted = " ".join(sorted(s1_cn.split()))
    t_sorted = " ".join(sorted(t_cn.split()))
    if s1_sorted == t_sorted and len(s1_cn.split()) >= 2:
        if effective_num_overlap and addr_tsr >= 60:
            return "R6_SORTED_ADDR_NUM", 0.95
        elif addr_tsr >= 75:
            return "R6_SORTED_ADDR75", 0.92

    s1_nosp = name_no_space(s1_cn)
    t_nosp = name_no_space(t_cn)
    if s1_nosp and t_nosp and s1_nosp == t_nosp and len(s1_nosp) >= 6:
        if effective_num_overlap and addr_tsr >= 60:
            return "R7_NOSP_ADDR_NUM", 0.94
        elif addr_tsr >= 78:
            return "R7_NOSP_ADDR78", 0.91

    name_tsort = fuzz.token_sort_ratio(s1_cn, t_cn)
    if name_tsort >= 95 and effective_num_overlap and addr_tsr >= 65:
        return "R8_TSORT95_NUM", 0.92
    if name_tsort >= 92 and addr_tsr >= 82:
        return "R8_TSORT92_ADDR82", 0.90

    if name_ratio >= 85 and addr_tsr >= 90:
        return "R9_NAME85_ADDR90", 0.92

    # R10: Substantial core similarity (no single-word geographic collisions)
    if effective_num_overlap and addr_tsr >= 85:
        if s1_cr and t_cr:
            if s1_cr == t_cr:
                return "R10_CORE_EXACT_ADDR85_NUM", 0.95
            core_tsr = fuzz.token_set_ratio(s1_cr, t_cr)
            if core_tsr >= 80:
                return "R10_CORE_TSR80_ADDR85_NUM", 0.93
        if fuzz.token_set_ratio(s1_cn, t_cn) >= 85:
            return "R10_NAME_TSR85_ADDR85_NUM", 0.92

    if (s1_cn in t_cn or t_cn in s1_cn) and len(min(s1_cn, t_cn)) >= 6:
        if effective_num_overlap and addr_tsr >= 75:
            return "R11_SUBSTR_ADDR_NUM", 0.94
        if addr_tsr >= 85:
            return "R11_SUBSTR_ADDR85", 0.92

    if (is_acronym(s1_cn, t_cn) or is_acronym(t_cn, s1_cn)) and effective_num_overlap and addr_tsr >= 75:
        return "R_ACRONYM_ADDR", 0.94

    # R12: Core name token_set_ratio (avoids 'private limited' false positives)
    if addr_tsr >= 92 and effective_num_overlap:
        if s1_cr and t_cr:
            core_tsr = fuzz.token_set_ratio(s1_cr, t_cr)
            if core_tsr >= 75:
                return "R12_ADDR92_CORE75_NUM", 0.92

    # R13: Distinctive first word + very high address match
    if addr_tsr >= 95 and not (s1_hnum and t_hnum and s1_hnum != t_hnum):
        s1_w0 = s1_cr.split()[0] if s1_cr else ""
        t_w0 = t_cr.split()[0] if t_cr else ""
        if s1_w0 and t_w0 and s1_w0 == t_w0 and len(s1_w0) >= 5 and s1_w0 not in COMMON_PREFIX_WORDS:
            return "R13_DISTINCT_ADDR95", 0.91

    return None, 0.0


def main():
    t_total_start = time.time()
    BATCH_SIZE = 500_000

    # Step 1: Load and Index Source 1
    print("\n[Step 1/3] Loading and indexing Source 1 queries...", flush=True)
    t0 = time.time()
    s1_df = pl.read_csv(s1_path, separator="\t")
    n_s1 = len(s1_df)

    s1_eids = s1_df["entity_id"].to_list()
    s1_raw_c = s1_df["country"].to_list()
    s1_raw_nm = s1_df["business_name"].to_list()
    s1_raw_ad = s1_df["business_address"].to_list()
    del s1_df
    gc.collect()

    s1_cc = [norm_country(x) for x in s1_raw_c]
    s1_cn = [clean_name(x) for x in s1_raw_nm]
    s1_ca = [clean_addr(s1_raw_ad[i], s1_cc[i]) for i in range(n_s1)]
    s1_cr = [core_name(x) for x in s1_cn]

    s1_nums, s1_hnums, s1_swords = [], [], []
    for i in range(n_s1):
        _, hnum, nums_set, swords, _ = extract_addr_features(s1_raw_ad[i], s1_ca[i], s1_cc[i])
        s1_nums.append(nums_set)
        s1_hnums.append(hnum)
        s1_swords.append(swords)

    index = defaultdict(list)
    for idx in range(n_s1):
        for k in extract_blocking_keys(s1_raw_c[idx], s1_raw_nm[idx], s1_raw_ad[idx], s1_ca[idx]):
            index[k].append(idx)

    MAX_BUCKET = {
        "N_EXACT": 100, "N_CORE": 50, "A_EXACT": 100,
        "N_FIRST2": 30, "N_SORTED": 30, "N_NOSP": 40,
        "N_OCR": 30, "ZIP_NUM": 30, "NUM_STREET": 25,
        "NA_NUM": 25,
    }
    for k in list(index.keys()):
        limit = MAX_BUCKET.get(k[0], 25)
        if len(index[k]) > limit:
            del index[k]

    print(f"[+] Loaded and indexed {n_s1:,} queries ({len(index):,} active keys) in {time.time()-t0:.1f}s.", flush=True)

    # Step 2: Stream Targets & Score
    print("\n[Step 2/3] Streaming targets & matching...", flush=True)
    matches_dict = defaultdict(list)

    def stream_and_match(path, label):
        t_src = time.time()
        print(f"[*] Processing {label}...", flush=True)
        reader = pl.read_csv_batched(path, separator="\t", batch_size=BATCH_SIZE)
        total_rows = 0
        total_comparisons = 0

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            b = batches[0]
            total_rows += len(b)
            eids = b["entity_id"].to_list()
            countries = b["country"].to_list()
            names = b["business_name"].to_list()
            addrs = b["business_address"].to_list()
            del b

            for eid, c, nm, ad in zip(eids, countries, names, addrs):
                norm_c = norm_country(c)
                cn = clean_name(nm)
                ca = clean_addr(ad, norm_c)
                cr = core_name(cn)

                matched_s1 = set()
                if cn:
                    k_exact = ("N_EXACT", norm_c, cn)
                    if k_exact in index:
                        matched_s1.update(index[k_exact])
                    if cr and cr != cn:
                        k_core = ("N_CORE", norm_c, cr)
                        if k_core in index:
                            matched_s1.update(index[k_core])
                    words = cn.split()
                    if len(words) >= 2:
                        k_f2 = ("N_FIRST2", norm_c, f"{words[0]} {words[1]}")
                        if k_f2 in index:
                            matched_s1.update(index[k_f2])
                        k_sort = ("N_SORTED", norm_c, " ".join(sorted(words)))
                        if k_sort in index:
                            matched_s1.update(index[k_sort])
                    nosp = name_no_space(cn)
                    if len(nosp) >= 5:
                        k_nosp = ("N_NOSP", norm_c, nosp)
                        if k_nosp in index:
                            matched_s1.update(index[k_nosp])

                postal, hnum, nums_set, street_words, multi_hnums = extract_addr_features(ad, ca, norm_c)
                for h in multi_hnums:
                    if postal and h:
                        k_zip = ("ZIP_NUM", norm_c, f"{postal}_{h}")
                        if k_zip in index:
                            matched_s1.update(index[k_zip])
                    if h and street_words:
                        k_ns = ("NUM_STREET", norm_c, f"{h}_{street_words[0]}")
                        if k_ns in index:
                            matched_s1.update(index[k_ns])
                    if h and cn:
                        w0 = cn.split()[0]
                        if len(w0) >= 3:
                            k_na = ("NA_NUM", norm_c, f"{h}_{w0}")
                            if k_na in index:
                                matched_s1.update(index[k_na])
                if ca:
                    k_a = ("A_EXACT", norm_c, ca)
                    if k_a in index:
                        matched_s1.update(index[k_a])

                if not matched_s1:
                    continue

                for s1_idx in matched_s1:
                    total_comparisons += 1
                    country_match = (s1_cc[s1_idx] == norm_c)
                    rule, conf = score_match(
                        s1_cn[s1_idx], s1_ca[s1_idx], s1_cr[s1_idx], s1_nums[s1_idx], s1_hnums[s1_idx], s1_swords[s1_idx], s1_raw_ad[s1_idx],
                        cn, ca, cr, nums_set, hnum, street_words, ad,
                        country_match, s1_cc[s1_idx], nm
                    )
                    if conf >= 0.88:
                        matches_dict[s1_idx].append((eid, conf))

            if total_rows % 1_000_000 == 0:
                print(f"     {label}: {total_rows:,} rows, {total_comparisons:,} comparisons ({time.time()-t_src:.1f}s)",
                      flush=True)

        print(f"[+] {label}: {total_rows:,} rows, {total_comparisons:,} comparisons in {time.time()-t_src:.1f}s.", flush=True)
        gc.collect()

    stream_and_match(s2_path, "Source 2")
    stream_and_match(s3_path, "Source 3")
    del index
    gc.collect()

    # Step 3: Output Generation
    print("\n[Step 3/3] Generating Output...", flush=True)
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "output/submmision"
    os.makedirs(output_dir, exist_ok=True)
    match_path = os.path.join(output_dir, "matching_results.tsv")
    match_v26 = os.path.join(output_dir, "matching_results_V26.tsv")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

    n_with_match = 0
    total_matches = 0

    with open(match_path, "w", encoding="utf-8") as f_match, open(cand_path, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for idx, eid in enumerate(s1_eids):
            m_list = matches_dict.get(idx, [])
            if m_list:
                seen = {}
                for mid, conf in m_list:
                    if mid not in seen or conf > seen[mid]:
                        seen[mid] = conf
                max_c = max(seen.values())

                if max_c >= 0.93:
                    top = [mid for mid, conf in sorted(seen.items(), key=lambda x: -x[1]) if conf >= max_c - 0.07][:6]
                elif max_c >= 0.90:
                    top = [mid for mid, conf in sorted(seen.items(), key=lambda x: -x[1]) if conf >= max_c - 0.05][:5]
                else:
                    top = []

                cands = [mid for mid, conf in sorted(seen.items(), key=lambda x: -x[1])][:12]
                for t_id in top:
                    if t_id not in cands:
                        cands.append(t_id)

                if top:
                    n_with_match += 1
                    total_matches += len(top)
                    f_match.write(f"{eid}\t{','.join(top)}\n")
                else:
                    f_match.write(f"{eid}\t\n")

                if cands:
                    f_cand.write(f"{eid}\t{','.join(cands)}\n")
                else:
                    f_cand.write(f"{eid}\t\n")
            else:
                f_match.write(f"{eid}\t\n")
                f_cand.write(f"{eid}\t\n")

            if (idx + 1) % 500_000 == 0:
                print(f"     Written {idx+1:,}/{n_s1:,} rows...", flush=True)

    shutil.copyfile(match_path, match_v26)
    size_mb = os.path.getsize(match_path) / (1024 * 1024)
    cand_mb = os.path.getsize(cand_path) / (1024 * 1024)
    total_time = time.time() - t_total_start
    match_pct = (n_with_match / n_s1 * 100) if n_s1 else 0
    avg_m = (total_matches / max(1, n_with_match))

    print("\n" + "=" * 80)
    print(f"[✔] Pipeline V26 Champion Successfully Completed in {total_time:.1f}s ({total_time/60:.1f} min)!")
    print(f"    Saved Matching   : {match_path} ({size_mb:.2f} MB)")
    print(f"    Saved V26        : {match_v26}")
    print(f"    Saved Candidates : {cand_path} ({cand_mb:.2f} MB)")
    print(f"    Entities Matched : {n_with_match:,} / {n_s1:,} ({match_pct:.1f}%)")
    print(f"    Singletons       : {n_s1 - n_with_match:,} ({(n_s1 - n_with_match)/n_s1*100:.1f}%)")
    print(f"    Total Matches    : {total_matches:,} (Avg: {avg_m:.2f}/entity)")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
