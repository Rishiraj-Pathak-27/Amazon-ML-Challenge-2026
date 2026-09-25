"""
Pairwise feature engineering for (Source-1, candidate) record pairs.
These features feed the LightGBM match / no-match classifier.
"""
import pandas as pd
from rapidfuzz import fuzz

from .normalize import (
    normalize_name,
    core_name,
    normalize_address,
    core_address,
    name_tokens,
    address_tokens,
    normalize_country,
)

FEATURE_COLUMNS = [
    "name_jaccard", "name_levenshtein", "name_token_sort", "name_partial",
    "addr_jaccard", "addr_levenshtein", "addr_token_sort", "country_match",
    "name_len_diff", "addr_len_diff", "common_name_tokens", "first_token_match",
    "exact_name_match", "exact_addr_match", "num_overlap",
]


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _extract_numbers(text):
    return set(tok for tok in text.split() if any(ch.isdigit() for ch in tok))


def compute_pair_features(s1_tuple, cand_tuple):
    """
    s1_tuple / cand_tuple: (business_name, business_address, country)
    """
    s1_name, s1_addr, s1_c = s1_tuple
    c_name, c_addr, c_c = cand_tuple

    s1_name_norm, c_name_norm = normalize_name(s1_name), normalize_name(c_name)
    s1_addr_norm, c_addr_norm = normalize_address(s1_addr), normalize_address(c_addr)

    s1_cname, c_cname = core_name(s1_name), core_name(c_name)
    s1_caddr, c_caddr = core_address(s1_addr), core_address(c_addr)

    s1_name_tok, c_name_tok = name_tokens(s1_name), name_tokens(c_name)
    s1_addr_tok, c_addr_tok = address_tokens(s1_addr), address_tokens(c_addr)

    s1_country = normalize_country(s1_c)
    c_country = normalize_country(c_c)

    s1_nums = _extract_numbers(s1_addr_norm)
    c_nums = _extract_numbers(c_addr_norm)
    num_match = int(bool(s1_nums and c_nums and (s1_nums & c_nums)))

    return {
        "name_jaccard": _jaccard(s1_name_tok, c_name_tok),
        "name_levenshtein": fuzz.ratio(s1_name_norm, c_name_norm) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(s1_name_norm, c_name_norm) / 100.0,
        "name_partial": fuzz.partial_ratio(s1_name_norm, c_name_norm) / 100.0,
        "addr_jaccard": _jaccard(s1_addr_tok, c_addr_tok),
        "addr_levenshtein": fuzz.ratio(s1_addr_norm, c_addr_norm) / 100.0,
        "addr_token_sort": fuzz.token_sort_ratio(s1_addr_norm, c_addr_norm) / 100.0,
        "country_match": int(s1_country == c_country and s1_country != ""),
        "name_len_diff": abs(len(s1_name_norm) - len(c_name_norm)),
        "addr_len_diff": abs(len(s1_addr_norm) - len(c_addr_norm)),
        "common_name_tokens": len(s1_name_tok & c_name_tok),
        "first_token_match": int(
            bool(s1_name_norm) and bool(c_name_norm)
            and s1_name_norm.split()[0] == c_name_norm.split()[0]
        ),
        "exact_name_match": int(bool(s1_cname) and s1_cname == c_cname),
        "exact_addr_match": int(bool(s1_caddr) and s1_caddr == c_caddr),
        "num_overlap": num_match,
    }


def build_feature_frame(pairs_df, source1_df, other_df):
    """
    pairs_df: DataFrame with columns [source1_entity_id, candidate_entity_id]
    All candidate_entity_id values must belong to `other_df`.
    Returns pairs_df with feature columns appended.
    """
    if len(pairs_df) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS)

    # Fast dict lookup only for required entities instead of all 5M records
    s1_needed = set(pairs_df["source1_entity_id"])
    s1_sub = source1_df[source1_df["entity_id"].isin(s1_needed)]
    s1_dict = dict(zip(
        s1_sub["entity_id"],
        zip(s1_sub["business_name"], s1_sub["business_address"], s1_sub["country"])
    ))

    other_needed = set(pairs_df["candidate_entity_id"])
    other_sub = other_df[other_df["entity_id"].isin(other_needed)]
    other_dict = dict(zip(
        other_sub["entity_id"],
        zip(other_sub["business_name"], other_sub["business_address"], other_sub["country"])
    ))

    records = []
    s1_ids = pairs_df["source1_entity_id"].to_numpy()
    cand_ids = pairs_df["candidate_entity_id"].to_numpy()

    for s1_id, cand_id in zip(s1_ids, cand_ids):
        s1_row = s1_dict.get(s1_id, ("", "", ""))
        cand_row = other_dict.get(cand_id, ("", "", ""))
        records.append(compute_pair_features(s1_row, cand_row))

    feat_df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    return pd.concat([pairs_df.reset_index(drop=True), feat_df], axis=1)
