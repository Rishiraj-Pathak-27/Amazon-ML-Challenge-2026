"""
Pairwise feature extraction between Source 1 and target candidates.
"""
from rapidfuzz import fuzz
from .normalize import is_acronym

FEATURE_NAMES = [
    "name_ratio",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_exact",
    "core_ratio",
    "core_token_set_ratio",
    "core_exact",
    "addr_token_set_ratio",
    "num_overlap",
    "hnum_match",
    "hnum_conflict",
    "country_match",
]


def compute_pairwise_features(s1_meta, t_meta):
    """
    Computes a 12-dimensional numerical feature vector for candidate pairs.
    s1_meta and t_meta are tuples: (clean_name, clean_addr, core_name, nums_set, hnum, norm_country)
    """
    s1_cn, s1_ca, s1_cr, s1_nums, s1_hnum, s1_c = s1_meta
    t_cn, t_ca, t_cr, t_nums, t_hnum, t_c = t_meta

    name_exact = 1.0 if (s1_cn and t_cn and s1_cn == t_cn) else 0.0
    name_ratio = 100.0 if name_exact else (fuzz.ratio(s1_cn, t_cn) if (s1_cn and t_cn) else 0.0)
    name_tsr = fuzz.token_set_ratio(s1_cn, t_cn) if (s1_cn and t_cn) else 0.0
    name_tsort = fuzz.token_sort_ratio(s1_cn, t_cn) if (s1_cn and t_cn) else 0.0

    core_exact = 1.0 if (s1_cr and t_cr and s1_cr == t_cr) else 0.0
    core_ratio = fuzz.ratio(s1_cr, t_cr) if (s1_cr and t_cr) else 0.0
    core_tsr = fuzz.token_set_ratio(s1_cr, t_cr) if (s1_cr and t_cr) else 0.0

    addr_tsr = fuzz.token_set_ratio(s1_ca, t_ca) if (s1_ca and t_ca) else 0.0
    num_overlap = 1.0 if (s1_nums and t_nums and (s1_nums & t_nums)) else 0.0

    hnum_match = 1.0 if (s1_hnum and t_hnum and s1_hnum == t_hnum) else 0.0
    hnum_conflict = 1.0 if (s1_hnum and t_hnum and s1_hnum != t_hnum) else 0.0
    country_match = 1.0 if (s1_c and t_c and s1_c == t_c) else 0.0

    return [
        name_ratio, name_tsr, name_tsort, name_exact,
        core_ratio, core_tsr, core_exact,
        addr_tsr, num_overlap, hnum_match, hnum_conflict, country_match
    ]
