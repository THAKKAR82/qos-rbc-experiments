# Decision: CFPB modeling gate

Date: 2026-10-16. Decided by: the project owner (user instruction for QOS applicability audit step 1).

## Decision

Modeling on the accepted CFPB v1 preparation [`20261014T192947Z-7b6668f9`](../../validation/cfpb/20261014T192947Z-7b6668f9/) proceeds under **Recommendation B** of the [curation validity review](../cfpb_curation_validity_review.md): the current curation is defensible with documented caveats.

The preparation's `release_gates.json` is **not edited** and still reads `modeling_permitted: false` with two `validity_review_required` entries. This record is the separate, explicit resolution of those entries. It does not change curation, thresholds, splits or the TF-IDF representation.

## Caveats that must accompany every result on this preparation

1. **`over_30_percent_class_removed:Debt collection`.** Curation removes 35.14% of Debt collection candidate rows (11,752 → 7,622), and its share of the primary population drops by 9.45 percentage points (46.26% → 36.81%). Removal rates are 33.82% / 35.22% / 38.84% across train / validation / test. Earliest-occurrence selection favors early observations, so this changes temporal prevalence even under a correct chronological split.
2. **`template_component_over_10_percent_class:Debt collection`.** The largest near-template component holds 1,576 Debt collection rows (13.4% of that class's candidates). It is a coherent, chained repeated-language family across 167 companies, not a verified duplicate incident. The primary keeps one January representative.

The primary estimand is performance on the earliest representatives of algorithmically defined text groups, not on the natural complaint population. The secondary natural-stream evaluation remains required wherever classifiers are compared.

## Prior use of this preparation

The accepted uncompressed baselines ([`20261014T204807Z-2270924a`](../../validation/cfpb/20261014T204807Z-2270924a/benchmark_report.md)) and the compression benchmark ([`20261016T125619Z-30526eac`](../../validation/cfpb/20261016T125619Z-30526eac/benchmark_report.md)) already used this preparation under Recommendation B. This record documents that resolution in one place. It does not retroactively change those runs.

## Referenced by

- `results/audit/cfpb_step1/` (manifest and every task JSON carry this path and its SHA-256)
- `notebooks/06_cfpb_qos_audit_step1.ipynb`
