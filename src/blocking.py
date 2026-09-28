"""
Multi-channel inverted index blocking for scalable candidate generation.
"""
from collections import defaultdict
from .normalize import norm_country, clean_name, core_name, name_no_space, clean_addr, extract_addr_features


def extract_blocking_keys(country, name, raw_addr, clean_a):
    norm_c = norm_country(country)
    cn = clean_name(name)
    keys = []
    if cn:
        keys.append(("N_EXACT", norm_c, cn))
        cr = core_name(cn)
        if cr and cr != cn:
            keys.append(("N_CORE", norm_c, cr))
        words = cn.split()
        if len(words) >= 2:
            keys.append(("N_FIRST2", norm_c, f"{words[0]} {words[1]}"))
            keys.append(("N_SORTED", norm_c, " ".join(sorted(words))))
        nosp = name_no_space(cn)
        if len(nosp) >= 5:
            keys.append(("N_NOSP", norm_c, nosp))

    postal, hnum, nums_set, street_words, multi_hnums = extract_addr_features(raw_addr, clean_a, norm_c)
    for h in multi_hnums:
        if postal and h:
            keys.append(("ZIP_NUM", norm_c, f"{postal}_{h}"))
        if h and street_words:
            keys.append(("NUM_STREET", norm_c, f"{h}_{street_words[0]}"))
        if h and cn:
            w0 = cn.split()[0]
            if len(w0) >= 3:
                keys.append(("NA_NUM", norm_c, f"{h}_{w0}"))
    if clean_a:
        keys.append(("A_EXACT", norm_c, clean_a))
    return keys


def build_inverted_index(entity_ids, countries, names, raw_addrs, clean_addrs, max_bucket_size=50):
    """Builds multi-channel inverted index mapping blocking keys to entity IDs."""
    index = defaultdict(list)
    for idx, (eid, c, nm, ad, ca) in enumerate(zip(entity_ids, countries, names, raw_addrs, clean_addrs)):
        for k in extract_blocking_keys(c, nm, ad, ca):
            index[k].append(idx)

    # Prune giant high-frequency buckets to prevent explosion
    max_buckets = {
        "N_EXACT": 100, "N_CORE": 50, "A_EXACT": 100,
        "N_FIRST2": 30, "N_SORTED": 30, "N_NOSP": 40,
        "ZIP_NUM": 30, "NUM_STREET": 25, "NA_NUM": 25,
    }
    for k in list(index.keys()):
        limit = max_buckets.get(k[0], max_bucket_size)
        if len(index[k]) > limit:
            del index[k]
    return index
