"""
Text normalization utilities for business names and addresses.

Handles common abbreviation expansion, punctuation stripping, and case
folding so that superficially different strings (different sources use
different conventions) compare fairly.
"""
import re

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Canonical expansion of common legal-entity abbreviations.
_NAME_ABBREVIATIONS = {
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "intl": "international",
    "mfg": "manufacturing",
    "assoc": "associates",
    "assocs": "associates",
    "bros": "brothers",
    "grp": "group",
    "svcs": "services",
    "svc": "service",
    "and": "and",
    "sa": "societe anonyme",
    "sarl": "societe a responsabilite limitee",
    "sas": "societe par actions simplifiee",
    "sasu": "societe par actions simplifiee unipersonnelle",
    "eurl": "entreprise unipersonnelle a responsabilite limitee",
    "sci": "societe civile immobiliere",
}

# Legal suffixes to drop when extracting the core business name
LEGAL_SUFFIXES = {
    "corporation", "corp", "company", "co", "incorporated", "inc",
    "limited", "ltd", "private", "pvt", "limited liability company", "llc",
    "limited liability partnership", "llp", "sa", "sarl", "sas", "sasu",
    "eurl", "sci", "group", "grp", "services", "svcs", "holdings",
    "enterprises", "trading", "industries",
}

# Canonical expansion of common address abbreviations.
_ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "bld": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "apt": "apartment",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "no": "number",
    "nr": "near",
    "opp": "opposite",
    "sq": "square",
    "hwy": "highway",
    "ste": "suite",
    "pk": "park",
    "ct": "court",
    "cir": "circle",
    "e": "east",
    "w": "west",
    "n": "north",
    "s": "south",
    "av": "avenue",
    "pl": "place",
    "rte": "route",
    "all": "allee",
}

# Filler / landmark words that carry little discriminative signal for
# address token-overlap features (kept in the normalized string, only
# excluded from the token-set used for Jaccard-style comparisons).
_ADDRESS_STOPWORDS = {"near", "opposite", "behind", "next", "to", "the", "of"}


def basic_clean(text):
    """Lowercase, expand '&', strip punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _expand_tokens(tokens, mapping):
    return [mapping.get(tok, tok) for tok in tokens]


def normalize_name(text):
    cleaned = basic_clean(text)
    tokens = _expand_tokens(cleaned.split(), _NAME_ABBREVIATIONS)
    return " ".join(tokens)


def core_name(text):
    norm = normalize_name(text)
    tokens = [tok for tok in norm.split() if tok not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_address(text):
    cleaned = basic_clean(text)
    tokens = _expand_tokens(cleaned.split(), _ADDRESS_ABBREVIATIONS)
    return " ".join(tokens)


def core_address(text):
    norm = normalize_address(text)
    tokens = [tok for tok in norm.split() if tok not in _ADDRESS_STOPWORDS]
    return " ".join(tokens)


def name_tokens(text):
    return set(normalize_name(text).split())


def address_tokens(text):
    toks = set(normalize_address(text).split())
    return toks - _ADDRESS_STOPWORDS


def normalize_country(text):
    if text is None:
        return ""
    return str(text).strip().lower()
