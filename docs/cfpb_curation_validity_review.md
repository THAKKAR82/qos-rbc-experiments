# CFPB curation validity review

Review date: 2026-10-14. Scope: frozen specification v1.0 and accepted text-stage run [`20261014T192947Z-7b6668f9`](../validation/cfpb/20261014T192947Z-7b6668f9/). No classifiers, TF-IDF refits, sketches, policy changes, or Bank changes were performed.

**Recommendation B: current curation is defensible with documented caveats. Proceed to baseline modeling only after user authorization, and require the already specified secondary natural-stream evaluation.** The primary estimand is performance on earliest representatives of eligible, algorithmically defined text groups—not performance on the natural complaint population, unique consumers, or independently verified incidents. The evidence supports template-family grouping, but does not support calling every removed row a duplicate complaint. Keep the frozen threshold; do not choose a replacement by retained population size.

The two review triggers reflect real selection effects. Debt collection loses 35.14% of its rows, and its population share drops by 9.45 percentage points. The largest component is a coherent repeated-language family with substantial chaining, not a clique. The heading filter also removes some legitimate free text. These restrictions must accompany subsequent results. This document is a recommendation, not a modification of `release_gates.json` or permission to train in this task.

## Evidence and audit method

Read [project policy](../AGENTS.md), the [frozen specification](cfpb_benchmark_spec.md), [curation implementation documentation](cfpb_curation.md), and the accepted audit, groups, edges, split/count tables, manifest and validation checks. The source SHA-256 is `664f1cae8d3a19dfd6894e2ca760f5b3f0bb4b71e466f38235473e1ea5b29f6d`.

Streamed source/audit CSVs in 100,000-row chunks, retaining the 34,351 narrative-bearing rows. Joined on complaint ID with one-to-one validation and verified every reconstructed cleaned-text hash against the accepted audit. No full document-feature matrix was constructed. All-product grouping precedes four-class selection, including conflicts involving excluded products.

Reviewed **48 components**, covering both interpretations of “largest near-template components”: the 20 largest overall and 20 largest containing Debt collection, including exact-only groups; additionally the 20 largest actual edge-connected components overall and containing Debt collection. Coverage includes **all seven components with at least 50 rows and all 17 multi-label near-template components**. Ties use stable component ID order. Component sizes count original complaint rows, not unique strings.

For every reviewed component, enumerated **all unordered pairs of distinct cleaned texts**, recalculated exact word-trigram Jaccard, and verified its saved edge intersections/unions against source text. Computed exact graph diameter by breadth-first searches from every node. Pair distributions include non-edges; medians are not restricted to qualifying edges. Also calculated original-row-weighted pair medians, including repeated texts. A single cleaned-text group has similarity 1 among its row pairs. Trigram sets discard order beyond three words, frequency and punctuation; similarity 1 need not imply string identity.

Representative prose and selected extreme pairs were inspected, with only paraphrases exported. This is not a human reading of every entire narrative and cannot establish author identity, truth of allegations, or campaign coordination. The heading audit checks all 872 rows structurally and inspects matched contexts in a deterministic, stratified **71-row sample**; this is not a probability estimate of the false-exclusion rate.

The [component appendix](../validation/cfpb/curation-review-20261014/component_review.md) gives each component's full ID, size, label composition, dates, company count, representative summary, grouping explanation, classification and pair statistics. The [machine-readable table](../validation/cfpb/curation-review-20261014/component_audit.csv) contains the same evidence. No original narratives or personal details are reproduced in these review outputs.

## Largest component: 1,576 Debt collection rows

Component `template-4df0d586aaf192fb5da76683fee7eace2dc6ad50b72ec6ae28767d9bc9afe540` is also identified in the appendix and CSV.

It contains **591 distinct original strings, 566 normalized strings and 505 distinct cleaned texts**, across **167 companies**, January 1–February 27. Its partitions are 939 train, 318 validation and 319 test rows. The primary retains one January representative and removes 1,575 later members: **38.14% of all Debt collection removals**.

Every row contains the distinctive phrases “falsely reporting on my credit” and “i have never had any accounts with them”, as well as “debt collector” and “against my rights”. The research-introduction phrase occurs in 1,575 rows; the unusual investigation/report clause in 1,497; the same unusual code token in 1,459. This repeated sentence sequence is much stronger evidence of shared wording than merely discussing credit reporting.

Representative meaning: a collector allegedly reports an account the writer never held; the narrative repeats that assertion, supplies an account/collector slot, and ends with a rights/investigation-code accusation. Observed variations include company names inserted at several positions, missing or malformed redaction tokens, account placeholders, punctuation, joined words, and omitted or damaged legal-code clauses. They are not exclusively substitutions of names/numbers. Removing standalone redactions alone reduces 566 normalized variants to 505 cleaned variants; near grouping performs the much larger consolidation. No claim is made that the 1,576 underlying complaints concern one event or consumer.

| Graph/pair measure | Observed value |
| --- | ---: |
| Distinct cleaned-text nodes | 505 |
| Verified edges | 19,769 |
| All distinct-text pairs | 127,260 |
| Pair minimum / median / maximum | 0.591549 / 0.803030 / 1.000000 |
| Distinct-text pairs below 0.85 | 84.47% |
| Original-row-weighted pair median | 0.841270 |
| Edge minimum / median / maximum | 0.850000 / 0.854839 / 1.000000 |
| Exact unweighted graph diameter | 6 edges |

A shortest path between the minimum-similarity endpoints has **four edges**, with similarities **0.852459 → 0.894737 → 0.852459 → 0.852459**, even though its endpoints score **0.591549**. Along this path, collector-name insertions and malformed redactions change, then the final code clause is shortened. The central accusation and sentence scaffold persist. Full node IDs, adjacent similarities and a diameter-realizing path are recorded in [chain evidence](../validation/cfpb/curation-review-20261014/largest_component_chain.json). This directly demonstrates chaining rather than assuming every member satisfies the threshold against a representative.

Within this component alone, the 0.95 graph has **399 components**, largest 186; the 0.90 graph has **279**, largest 307; the frozen 0.85 graph has one, size 1,576. Its connectivity is therefore threshold-sensitive. That does not by itself imply a mistaken semantic merge: the checked endpoints still share the same distinctive template scaffold. Grouping the family as one **potential text-leakage group** is defensible for a conservative group-disjoint experiment. Treating it as one verified duplicate incident would not be defensible. Independent legitimate complaints may use the same form.

The largest company contributions are 173, 162, 132, 120 and 91 rows; no company dominates the component. Weekly counts are 89, 191, 121, 321, 217, 204, 114, 195 and 124, using Monday–Sunday bins clipped to the source period. This is repeated use throughout the period, not one isolated day. [Company](../validation/cfpb/curation-review-20261014/largest_component_companies.csv) and [temporal](../validation/cfpb/curation-review-20261014/largest_component_temporal.csv) tables provide details. Template dissemination or a submission campaign is plausible, but coordinated origin is **not established**.

## Other large and multi-label components

All seven components with at least 50 rows are Debt collection:

| Rows | Cleaned variants | Companies | Dates | Pair min / median / max | Interpretation |
| ---: | ---: | ---: | --- | --- | --- |
| 1,576 | 505 | 167 | Jan 1–Feb 27 | .591549 / .803030 / 1 | Shared false-reporting template family; substantial chaining |
| 596 | 5 | 3 | Jan 6–Feb 25 | .928571 / .963636 / 1 | Almost identical inaccurate-entry verification/deletion request |
| 235 | 85 | 66 | Feb 17–26 | .680272 / .850185 / 1 | Ordered debt-validation letter; redaction/truncation variants |
| 89 | 12 | 45 | Jan 1–Feb 28 | .717391 / .843074 / .987805 | Validation/cease-collection legal template |
| 81 | 4 | 42 | Jan 1–Feb 17 | .837838 / .913391 / 1 | Validation-information and inaccurate-furnishing boilerplate |
| 71 | 1 | 42 | Jan 1–Feb 5 | 1 / 1 / 1 | Identical cleaned validation letter |
| 50 | 4 | 31 | Jan 1–Feb 19 | .793333 / .894722 / .932432 | Ordered validation-document request |

These are stronger matches than generic topic overlap, but many can represent separate legitimate complaints using common language. Explicit form fields occur in some smaller components. Mass-submission campaigns cannot be confirmed from public prose and dates alone.

The **17 multi-label components contain 65 rows** and are fully enumerated in the appendix. Some are legal/identity-theft templates spanning products. Others are detailed repeated-event narratives: for example, a lender/bank payment dispute, a card/deposit-account subscription dispute, or repeated warranty-claim follow-up. Their labels can describe different sides of an event; “conflict” is an operational grouping designation, not proof of erroneous labels. Whole-group quarantine remains defensible because neither selecting a preferred class nor majority voting is justified. No labels were changed.

The frozen rule also leaves semantically similar families separate when no 0.85 path exists; appendix component 45 resembles the largest family. Thus **group-disjoint does not guarantee all semantic template families are disjoint**, and the audit is not a proof of zero residual text leakage.

## Audit-only threshold sensitivity

The saved complete 0.85 edge set contains every possible stricter-threshold edge. Filtered using integer intersection/union comparisons, rebuilt connected components across all products, recomputed component conflicts, then retained actual earliest `(date, integer ID)` representatives without moving any future example into training. Preserved artifact rules and earlier raw/model conflict quarantines. The 0.85 result matches the accepted primary **ID set exactly**, not just its count.

Two exact controls avoid ambiguity: **raw-normalized only** groups on the lowercase/whitespace-normalized key; **cleaned exact only** also collapses identical post-redaction model text, as v1 requires. Both keep mandatory earlier conflict/artifact quarantines fixed. Neither is a replacement policy.

| Threshold/control | Total N | Debt | Card | Checking | Mortgage | Largest component | Cross-split groups | New conflicting groups (rows) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw-normalized only | 22,168 | 9,013 | 5,517 | 5,606 | 2,032 | 581 | 199 | 0 (0) |
| Cleaned exact only | 22,086 | 8,935 | 5,514 | 5,605 | 2,032 | 581 | 197 | 0 (0) |
| Jaccard ≥ .95 | 21,611 | 8,480 | 5,502 | 5,597 | 2,032 | 596 | 209 | 7 (28) |
| Jaccard ≥ .90 | 21,198 | 8,096 | 5,480 | 5,592 | 2,030 | 596 | 214 | 14 (46) |
| Jaccard ≥ .85 | 20,705 | 7,622 | 5,472 | 5,584 | 2,027 | 1,576 | 134 | 17 (65) |

Largest/cross-split/conflict counts describe the all-product grouping universe before representative selection and final component-conflict removal. Cross-split means a group touches two or more of the fixed partitions, not that retained primary groups overlap. Counts can fall when multiple cross-split groups merge. Earlier **11 normalized conflicting groups / 46 rows** and **one model-text conflicting group / two rows** remain quarantined in every row of this table; exact-original conflicts overlap normalized conflicts and must not be added again. “New conflicting” therefore does not mean the dataset has no other conflicts.

| Threshold/control | Debt removed | Card removed | Checking removed | Mortgage removed |
| --- | ---: | ---: | ---: | ---: |
| Raw-normalized only | 2,739 | 254 | 165 | 76 |
| Cleaned exact only | 2,817 | 257 | 166 | 76 |
| ≥ .95 | 3,272 | 269 | 174 | 76 |
| ≥ .90 | 3,656 | 291 | 179 | 78 |
| ≥ .85 | 4,130 | 299 | 187 | 81 |

Compared with cleaned exact only, near grouping removes another **1,381** primary rows, **1,313 of them Debt collection**. From .90 to .85, Debt loses another 474 rows while the other classes together lose 19. Population and connectivity are meaningfully sensitive; much Debt removal nevertheless already occurs without near grouping. No threshold was selected using these retention figures. See [sensitivity CSV](../validation/cfpb/curation-review-20261014/threshold_sensitivity.csv).

## Structured-heading exclusions

All **872** mutually exclusive heading exclusions were checked for their triggering matches. There are 525 with one match and 347 with multiple matches; 291 have an explicit product/sub-product heading, while 538 have only issue/company matches. These categories are descriptive and not estimates of proven leakage. The primary four classes contribute **616 exclusions (2.425% of raw primary rows)**: Debt 188, Card 203, Checking 151, Mortgage 74. The other 256 belong to non-primary products.

Representative structures, paraphrased without case details:

- `Company: [institution] Product: [category] Issue: [problem]` followed by a complaint narrative: clearly structured, consistent with form/copy-generated drafting, but its origin is unverified.
- `Product: [product description]` followed by free prose: explicit product artifact, without proof it was copied from the dataset label.
- `Summary of the issue:` or `Description of the issue:` followed by an event history: legitimate narrative organization, not necessarily label leakage.
- A sentence ending `resolve this issue:` followed by explanation: a grammatical colon triggers the same rule.
- A prior `Complaint ID:` embedded in an ongoing dispute, or `Company response:` introducing a quoted response: meaningful prose can be removed along with the marker.

The inspected sample contains concrete counterexamples to interpreting the rule as a precise leakage detector. Some ordinary free text is definitely removed by the rule's broad surface pattern; it is not possible to count all such cases from regex counts alone. The 71 sampled matched contexts were chosen across heading types and single/multiple matches; they are not a random prevalence survey. [All-row heading audit](../validation/cfpb/curation-review-20261014/heading_audit.csv) and [sample IDs](../validation/cfpb/curation-review-20261014/heading_manual_sample.csv) permit private follow-up without publishing narratives.

**Judgment:** the rule is reproducible and sufficiently explicit for the specification's conservative artifact-excluded population, but not precise enough to describe all 872 exclusions as leakage or all as form-generated. This is a substantive coverage caveat, explicitly anticipated by v1. Keep it frozen in this review. A future claim about unrestricted consumer prose would require a separately specified heading-sensitivity analysis; the existing natural-stream population does not restore these rows.

## Class-distribution impact

“Raw” below means eligible source narratives with one of the four exact product labels, before curation—not all 1,038,383 complaints.

| Class | Raw | Curated | Loss | Loss % | Raw share % | Curated share % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Debt collection | 11,752 | 7,622 | 4,130 | 35.14 | 46.26 | 36.81 |
| Credit card | 5,771 | 5,472 | 299 | 5.18 | 22.72 | 26.43 |
| Checking or savings account | 5,771 | 5,584 | 187 | 3.24 | 22.72 | 26.97 |
| Mortgage | 2,108 | 2,027 | 81 | 3.84 | 8.30 | 9.79 |
| Total | 25,402 | 20,705 | 4,697 | 18.49 | 100 | 100 |

Debt removals comprise 3,911 later group members, 188 headings, 15 raw conflicts, one model conflict, 14 template conflicts and one empty cleaned text. **94.70% of Debt loss is later-group removal.** The dominant effect is deliberate template-frequency suppression, not an implementation failure or the heading rule.

| Class | Train raw → curated | Validation raw → curated | Test raw → curated | Within-class split proportions, raw → curated (train / validation / test %) |
| --- | --- | --- | --- | --- |
| Debt | 7,375 → 4,881 | 1,766 → 1,144 | 2,611 → 1,597 | 62.76 / 15.03 / 22.22 → 64.04 / 15.01 / 20.95 |
| Card | 3,788 → 3,591 | 819 → 782 | 1,164 → 1,099 | 65.64 / 14.19 / 20.17 → 65.63 / 14.29 / 20.08 |
| Checking | 3,792 → 3,674 | 835 → 806 | 1,144 → 1,104 | 65.71 / 14.47 / 19.82 → 65.80 / 14.43 / 19.77 |
| Mortgage | 1,372 → 1,313 | 315 → 301 | 421 → 413 | 65.09 / 14.94 / 19.97 → 64.78 / 14.85 / 20.37 |

There **is major class-specific sampling distortion relative to the raw narrative population**. Macro-F1 and balanced accuracy do not undo within-class selection. Results can be defensible for the declared group-representative task while being unrepresentative of incoming complaint frequency. All classes still exceed the frozen minimum per-partition support gates. [Class/split counts](../validation/cfpb/curation-review-20261014/class_split_impact.csv) and [exclusion reasons](../validation/cfpb/curation-review-20261014/exclusion_reasons.csv) provide exact denominators.

## Temporal and company impact

Overall removal rates are **17.57% train, 18.80% validation and 21.10% test**. For Debt they are **33.82%, 35.22% and 38.84%**. Earliest-occurrence selection predictably favors early observations; no future row was reassigned backward. This changes temporal prevalence even with correct chronological splitting.

| Week starting (clipped to source coverage) | Raw primary | Curated | All-class loss % | Debt loss % |
| --- | ---: | ---: | ---: | ---: |
| Dec 29 (Jan 1–4 only) | 1,698 | 1,376 | 18.96 | 37.29 |
| Jan 5 | 3,348 | 2,800 | 16.37 | 32.50 |
| Jan 12 | 2,779 | 2,291 | 17.56 | 33.83 |
| Jan 19 | 4,571 | 3,788 | 17.13 | 31.60 |
| Jan 26 | 3,955 | 3,227 | 18.41 | 35.71 |
| Feb 2 | 2,284 | 1,842 | 19.35 | 35.43 |
| Feb 9 | 1,660 | 1,381 | 16.81 | 33.24 |
| Feb 16 | 3,399 | 2,736 | 19.51 | 36.43 |
| Feb 23 (through Feb 28) | 1,708 | 1,264 | 26.00 | 45.65 |

The last week's Debt loss is **404/885**. Of these, 144 belong to the 235-row validation-letter component, 124 to the largest component and 41 to the 596-row component: **309/404 (76.49%)** are explained by three known templates. The 235-row component first occurs February 17 and retains its representative in test; it is not moved into train. This explains a substantial late-period change, not an unexplained parser/date artifact. Weekly bins cross some split boundaries; the separate split table uses the exact frozen boundaries.

Company effects are also large. Among the four-class raw narratives, Experian loses 330/481 (68.61%), TransUnion 296/448 (66.07%), Equifax 198/325 (60.92%), CL Holdings 322/823 (39.13%), Encore 312/711 (43.88%), and CCS 191/346 (55.20%). These are source-selected product labels, including Debt complaints against reporting companies; no labels were remapped. Company names are descriptive audit fields, not added predictors. [Company totals](../validation/cfpb/curation-review-20261014/company_impact.csv), [company-by-class](../validation/cfpb/curation-review-20261014/company_class_impact.csv) and [week-by-class](../validation/cfpb/curation-review-20261014/week_class_impact.csv) expose these concentrations.

The curated temporal/company distribution therefore **does introduce selection effects**. They are consistent with the explicit earliest-template policy rather than hidden splitting logic, but cannot be dismissed as harmless. Do not interpret temporal or company-level performance differences as pure model drift without acknowledging this curation.

## Conditions attached to recommendation B

1. Describe the primary population as artifact-eligible earliest text-group representatives; never claim deduplicated consumers, unique incidents, or natural-stream class frequencies.
2. Require the frozen secondary evaluation on the **same primary-trained models**, with seen-in-training/seen-in-validation/new-in-test strata. It already preserves **3,634 validation and 5,214 test rows**, versus primary 3,033 and 4,213. Debt supports become 1,730 and 2,563, respectively. This directly exposes the consequence of template repetition without changing the primary threshold or tuning on test.
3. Report primary and natural-stream per-class results together. Natural-stream evaluation remains artifact/conflict-filtered and cannot repair the heading-selection bias or all raw-population coverage limitations.
4. Carry the class, week, company and threshold-sensitivity tables into benchmark interpretation. Do not select a threshold or change the policy using future labels or classifier scores. A less-aggressive policy comparison, if later desired, needs separate predeclared authorization and must not replace an unfavorable primary result.
5. Retain the distinction between text-leakage groups and real-world complaint independence. Global future text/labels inform offline curation; this is not a prospective online deduplication algorithm.

**Why B, rather than A or C:** repetition is empirically real and the largest connected family has a recognizable shared scaffold, supporting conservative leakage grouping. Exact-only and stricter controls confirm that substantial Debt concentration is not created entirely by the 0.85 threshold. However, the magnitude and concentration of removal, chaining sensitivity, legitimate heading exclusions and late-period suppression rule out an unqualified endorsement. No evidence found here compels changing the frozen primary rule for its narrowly declared estimand. If the intended claim instead becomes performance on representative incoming CFPB complaints, this curated population alone is inadequate and the design must be reconsidered before that claim is made.

## Validation and limits

Audit checks confirmed source identity, source/audit one-to-one joins, all 34,351 model-text hashes, coverage of every required component category, reviewed saved-edge intersections/unions, exhaustive within-component pair statistics, graph diameters, and exact reproduction of accepted primary IDs at 0.85. Protected-file hashes were checked before and after; the [review manifest](../validation/cfpb/curation-review-20261014/review_manifest.json) records results and output hashes.

No classifier results exist from this review; no predictive scientific conclusions can yet be drawn. No fast/scientific test suite or full Bank sweep was rerun because no implementation, tests, dependencies or Bank inputs changed; accepted test results are historical, not claimed as newly executed. This review performs independent data/count checks only. TF-IDF statistics and theoretical QOS estimates were not recomputed. They remain conditional accounting, not measured quantum hardware usage or evidence of quantum advantage.
