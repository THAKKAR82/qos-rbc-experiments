"""Offline CFPB v1 curation, independent of classifiers and Bank experiments.

All products participate in conflict/template auditing. Only the frozen four
classes enter the benchmark. Global grouping is offline curation, not a claim
that future text or labels would be available in an online deployment.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re

import pandas as pd


SPEC_VERSION = "1.0"
SOURCE_SHA256 = "664f1cae8d3a19dfd6894e2ca760f5b3f0bb4b71e466f38235473e1ea5b29f6d"
CLASSES = ("Debt collection", "Credit card", "Checking or savings account", "Mortgage")
SOURCE_COLUMNS = ("Complaint ID", "Date received", "Product", "Consumer complaint narrative")
SPLITS = ("train", "validation", "test")
HEADING = re.compile(
    r"(?<!\w)(?:product|sub[ -]?product|issue|sub[ -]?issue|company|"
    r"company response(?: to consumer)?|complaint id|date received|"
    r"date sent to company|submitted via)\s*:", re.IGNORECASE,
)
REDACTION = re.compile(r"\bx{2,}\b")
MODEL_TOKEN = re.compile(r"(?u)\b\w\w+\b")
WORD = re.compile(r"(?u)\b\w+\b")
GROUP_COLUMNS = ["kind", "group_id", "member_count", "label_count", "labels",
                 "date_min", "date_max", "partition_count", "representative_id",
                 "conflicting", "disposition"]
EDGE_COLUMNS = ["left_model_group", "right_model_group", "intersection", "union",
                "jaccard", "component_id"]


def normalize_narrative(value):
    """Frozen lowercase/Unicode-whitespace normalization; no Unicode folding."""
    if value is None or pd.isna(value):
        return ""
    if not isinstance(value, str):
        raise TypeError("Narratives must be strings or missing values")
    return re.sub(r"\s+", " ", value.lower()).strip()


def clean_narrative(value):
    """Remove standalone redaction runs only; callers quarantine headings."""
    return re.sub(r"\s+", " ", REDACTION.sub(" ", normalize_narrative(value))).strip()


def chronological_split(date):
    """Return a frozen split for an ISO calendar date, or raise on invalid input."""
    if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError("Date must be an ISO calendar date")
    pd.Timestamp(date)  # Reject invalid calendar dates.
    if not "2026-01-01" <= date <= "2026-02-28":
        raise ValueError("Date is outside the frozen CFPB period")
    return "train" if date <= "2026-01-31" else (
        "validation" if date <= "2026-02-14" else "test"
    )


def source_metadata(frame):
    """Validate a chunk; invalid IDs/dates are audited, out-of-range dates block.

    The caller must additionally check ID uniqueness across all source chunks.
    No numeric coercion through floating point is used for complaint IDs.
    """
    absent = set(SOURCE_COLUMNS) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing source columns: {sorted(absent)}")
    rows = []
    for cid, date, product, narrative in frame.loc[:, list(SOURCE_COLUMNS)].itertuples(index=False, name=None):
        cid = "" if pd.isna(cid) else str(cid)
        date = "" if pd.isna(date) else str(date)
        product = "" if pd.isna(product) else str(product)
        invalid_id = not bool(re.fullmatch(r"[+-]?\d+", cid))
        parsed_id = "" if invalid_id else str(int(cid))
        invalid_date = not bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date))
        if not invalid_date:
            try:
                pd.Timestamp(date)
            except (ValueError, OverflowError):
                invalid_date = True
        split = "" if invalid_date else chronological_split(date)
        unavailable = not normalize_narrative(narrative)
        reason = ("invalid_id" if invalid_id else "invalid_date" if invalid_date
                  else "empty_narrative" if unavailable else "")
        rows.append(dict(source_complaint_id=cid, complaint_id=parsed_id,
                         date_received=date, product=product, split=split,
                         invalid_id=invalid_id, invalid_date=invalid_date,
                         empty_narrative=unavailable, reason=reason))
    return pd.DataFrame(rows, columns=["source_complaint_id", "complaint_id", "date_received",
                                      "product", "split", "invalid_id", "invalid_date",
                                      "empty_narrative", "reason"])


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _group_id(kind, rows):
    members = sorted(int(r["complaint_id"]) for r in rows)
    payload = json.dumps([SPEC_VERSION, members], separators=(",", ":"))
    return kind + "-" + _digest(payload)


def word_trigrams(text):
    """Collision-free word-trigram tuples; punctuation separates words."""
    tokens = WORD.findall(text)
    return frozenset(zip(tokens, tokens[1:], tokens[2:]))


def exact_jaccard_edges(sets, *, check_resources=None):
    """All Jaccard >=17/20 edges, with >=20 tokens in each set.

    Lossless prefix join: globally order tokens by document frequency, then
    lexical order. A qualifying pair must share a token in prefixes of length
    n-ceil(17*n/20)+1. Otherwise the earliest common token would leave too many
    nonmatching tokens before it. Index *all* prefixes; never cap candidates.
    Ascending set sizes allow the necessary length bound 20*m >=17*n.
    Every candidate is verified using exact sets and integer arithmetic.

    Returns (left_index, right_index, intersection, union) plus diagnostics.
    Input sets are never mutated. This is not approximate LSH or hash equality.
    """
    frequency = Counter(token for s in sets for token in s if len(s) >= 20)
    order = sorted((i for i, s in enumerate(sets) if len(s) >= 20),
                   key=lambda i: (len(sets[i]), i))
    postings = defaultdict(list)
    edges = []
    candidates_checked = 0
    for iteration, i in enumerate(order):
        if check_resources and iteration % 100 == 0:
            check_resources()
        s = sets[i]
        prefix_length = len(s) - (17 * len(s) + 19) // 20 + 1
        prefix = sorted(s, key=lambda token: (frequency[token], token))[:prefix_length]
        candidates = set()
        for token in prefix:
            candidates.update(j for j in postings[token] if 20 * len(sets[j]) >= 17 * len(s))
        for j in sorted(candidates):
            candidates_checked += 1
            intersection = len(s.intersection(sets[j]))
            union = len(s) + len(sets[j]) - intersection
            if 20 * intersection >= 17 * union:
                edges.append((min(i, j), max(i, j), intersection, union))
        for token in prefix:
            postings[token].append(i)
    return sorted(edges), dict(complete=True, method="lossless_global_order_prefix_join",
                               minimum_trigrams=20, candidates_checked=candidates_checked,
                               compared_texts=len(order), short_texts=len(sets) - len(order))


@dataclass
class CurationResult:
    audit: pd.DataFrame
    groups: pd.DataFrame
    edges: pd.DataFrame
    model_texts: dict
    diagnostics: dict


def curate_cfpb(frame, *, check_resources=None):
    """Curate all-product input without mutation; retain only primary classes.

    Full-file callers may stream empty/invalid rows separately, then call this
    on all valid narrative-bearing rows, after global source-ID validation.
    """
    meta = source_metadata(frame)
    if meta.empty:
        raise ValueError("No source rows supplied")
    ids = meta.loc[~meta.invalid_id, "complaint_id"]
    if ids.duplicated().any():
        raise ValueError("Duplicate complaint IDs block curation")
    rows = meta.to_dict("records")
    flags = ("structured_heading", "empty_model_text", "missing_product",
             "raw_conflict", "model_conflict", "template_conflict")
    for r, original in zip(rows, frame["Consumer complaint narrative"]):
        norm = normalize_narrative(original)
        model = clean_narrative(original)
        r.update({k: False for k in flags})
        r.update(raw_hash=_digest(original) if isinstance(original, str) else "",
                 normalized_hash=_digest(norm), model_text_hash=_digest(model),
                 exact_group_id="", normalized_group_id="", model_group_id="",
                 template_group_id="", representative_id="", component_size=0,
                 primary=False, natural_stream=False, stratum="", label=-1,
                 characters=len(original) if isinstance(original, str) else 0,
                 whitespace_tokens=len(original.split()) if isinstance(original, str) else 0,
                 language_review="not_assessed", _original=original, _norm=norm, _model=model)
        r["structured_heading"] = bool(HEADING.search(norm))
        r["empty_model_text"] = not bool(MODEL_TOKEN.search(model))
        r["missing_product"] = not bool(r["product"].strip())
    rows.sort(key=lambda r: (r["date_received"], int(r["complaint_id"]) if r["complaint_id"] else -1,
                             r["source_complaint_id"]))
    records = []

    def register(kind, members):
        gid = _group_id(kind, members)
        labels = sorted({r["product"] for r in members if r["product"].strip()})
        first = members[0]  # Members inherit deterministic chronological order.
        record = dict(kind=kind, group_id=gid, member_count=len(members), label_count=len(labels),
                      labels=json.dumps(labels), date_min=first["date_received"],
                      date_max=members[-1]["date_received"],
                      partition_count=len({r["split"] for r in members}),
                      representative_id=first["complaint_id"], conflicting=len(labels) > 1,
                      disposition="conflict" if len(labels) > 1 else "nonconflicting")
        records.append(record)
        for r in members:
            r[kind + "_group_id"] = gid
        return record

    def group_by(key, candidates):
        groups = defaultdict(list)
        for r in candidates:
            groups[r[key]].append(r)
        return list(groups.values())

    valid = [r for r in rows if not r["reason"]]
    for members in group_by("_original", valid):
        register("exact", members)
    for members in group_by("_norm", valid):
        group = register("normalized", members)
        for r in members:
            r["raw_conflict"] = group["conflicting"]

    candidates = [r for r in valid if not any(r[k] for k in
                  ("raw_conflict", "structured_heading", "empty_model_text", "missing_product"))]
    model_groups = []
    for members in group_by("_model", candidates):
        group = register("model", members)
        for r in members:
            r["model_conflict"] = group["conflicting"]
        if not group["conflicting"]:
            model_groups.append(members)
    model_groups.sort(key=lambda members: members[0]["model_group_id"])
    sets = [word_trigrams(members[0]["_model"]) for members in model_groups]
    links, diagnostics = exact_jaccard_edges(sets, check_resources=check_resources)
    del sets
    parent = list(range(len(model_groups)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _, _ in links:
        left, right = root(i), root(j)
        parent[max(left, right)] = min(left, right)
    components = defaultdict(list)
    for i, members in enumerate(model_groups):
        components[root(i)].extend(members)
    for members in components.values():
        members.sort(key=lambda r: (r["date_received"], int(r["complaint_id"])))
        group = register("template", members)
        first = members[0]
        for r in members:
            r["template_conflict"] = group["conflicting"]
            r["component_size"] = len(members)
            r["representative_id"] = first["complaint_id"]
            if not group["conflicting"] and r["product"] in CLASSES:
                r["primary"] = r is first
                r["natural_stream"] = r["split"] != "train"
                r["stratum"] = ("seen_in_train" if first["split"] == "train" else
                                "seen_in_validation" if first["split"] == "validation" else
                                "first_seen_in_test")
    edge_rows = [dict(left_model_group=model_groups[i][0]["model_group_id"],
                      right_model_group=model_groups[j][0]["model_group_id"],
                      intersection=inter, union=union, jaccard=inter / union,
                      component_id=model_groups[i][0]["template_group_id"])
                 for i, j, inter, union in links]
    texts = {}
    for r in rows:
        if r["product"] in CLASSES:
            r["label"] = CLASSES.index(r["product"])
        if not r["reason"]:
            r["reason"] = next((key for key in ("missing_product", "raw_conflict",
                                "structured_heading", "empty_model_text", "model_conflict",
                                "template_conflict") if r[key]), "")
            if not r["reason"]:
                r["reason"] = ("nonprimary_class" if r["product"] not in CLASSES else
                               "included_primary" if r["primary"] else "later_group_member")
        if r["primary"] or r["natural_stream"]:
            texts[r["complaint_id"]] = r["_model"]
        for key in ("_original", "_norm", "_model"):
            del r[key]
    audit = pd.DataFrame(rows)
    groups = pd.DataFrame(records, columns=GROUP_COLUMNS).sort_values(["kind", "group_id"])
    edges = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS).sort_values(
        ["left_model_group", "right_model_group"])
    diagnostics["near_template_components"] = len({root(i) for i, j, _, _ in links})
    diagnostics["template_components_including_singletons"] = len(components)
    return CurationResult(audit, groups.reset_index(drop=True), edges.reset_index(drop=True),
                          texts, diagnostics)


def curation_gates(audit):
    """Frozen population gates; review gates prohibit models, not text profiling."""
    primary = audit[audit.primary]
    counts = pd.crosstab(primary["product"], primary["split"]).reindex(
        index=CLASSES, columns=SPLITS, fill_value=0)
    blocking = []
    review = []
    if counts.train.sum() < 1000:
        blocking.append("fewer_than_1000_training_rows")
    for product in CLASSES:
        for split, minimum in (("train", 200), ("validation", 100), ("test", 100)):
            if counts.loc[product, split] < minimum:
                blocking.append(f"insufficient_class_support:{product}:{split}")
        raw = audit[(audit["product"] == product) & ~audit.empty_narrative]
        if len(raw) and 1 - len(primary[primary["product"] == product]) / len(raw) > .30:
            review.append(f"over_30_percent_class_removed:{product}")
        component_counts = raw[raw.template_group_id.ne("")].groupby("template_group_id").size()
        if len(raw) and len(component_counts) and component_counts.max() > .10 * len(raw):
            review.append(f"template_component_over_10_percent_class:{product}")
    if primary.template_group_id.duplicated().any():
        blocking.append("primary_group_overlap")
    return dict(blocking=blocking, validity_review_required=review,
                modeling_permitted=not blocking and not review)
