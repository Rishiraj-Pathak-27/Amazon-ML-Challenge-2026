"""
Text normalization, phonetic/abbreviation mapping, and structural feature extractors.
"""
import re

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
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth",
}


def norm_country(c):
    if not c or c != c:
        return ""
    c_clean = str(c).strip().lower()
    return _COUNTRY_MAP.get(c_clean, c_clean)


def clean_name(name):
    if not name or name != name:
        return ""
    s = str(name).lower()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"https?://\S+|www\.\S+|\.(?:com|org|net|in|fr|co|io|biz|info|gov)\b", "", s)
    s = re.sub(r"[-–—]\s*\d{6,}\b", "", s)
    s = re.sub(r"#\s*\d+\b", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def core_name(cleaned_name):
    if not cleaned_name:
        return ""
    words = cleaned_name.split()
    filtered = [w for w in words if w not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else cleaned_name


def name_no_space(cleaned_name):
    if not cleaned_name:
        return ""
    return re.sub(r"\s+", "", cleaned_name)


def clean_addr(addr, country):
    if not addr or addr != addr:
        return ""
    s = str(addr).lower()
    s = re.sub(r"\b(?:null|none|nan|n\s*/\s*a|<null>)\b", " ", s)
    s = re.sub(r"[^\w\s/]", " ", s)
    tokens = s.split()
    expanded = [_ADDR_COMMON.get(t, t) for t in tokens]
    return " ".join(expanded)


def extract_addr_features(raw_addr, clean_a, country):
    postal = ""
    if country == "us":
        m = re.findall(r"\b(\d{5})(?:-\d{4})?\b", str(raw_addr))
        if m:
            postal = m[-1]
    elif country == "in":
        m = re.findall(r"\b([1-9]\d{5})\b", str(raw_addr))
        if m:
            postal = m[-1]
    elif country == "fr":
        m = re.findall(r"\b(\d{5})\b", str(raw_addr))
        if m:
            postal = m[-1]

    nums = re.findall(r"\b\d+\b", clean_a)
    hnum = nums[0] if nums else ""
    nums_set = set(nums)

    street_words = [w for w in clean_a.split() if len(w) >= 4 and not w.isdigit()]
    multi_hnums = nums[:3]
    return postal, hnum, nums_set, street_words, multi_hnums


def is_acronym(short_name, long_name):
    if not short_name or not long_name:
        return False
    words = long_name.split()
    if len(words) < 2 or len(short_name) != len(words):
        return False
    stop = {"of", "the", "and", "for", "in", "on", "at", "to", "a", "an"}
    words_nostop = [w for w in words if w not in stop]
    acr_nostop = "".join(w[0] for w in words_nostop if w)
    return short_name == acr_nostop
