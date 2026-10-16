# large_dataset_mode

Use when the entrypoint's size, RAM-pressure, or user-request trigger applies. This changes profiling and recommendations, not permission to modify data/code or start experiments. Preserve all target, leakage, split, and QOS-accounting rules in `SKILL.md` and `AGENTS.md`.

## Inspect and budget before loading

Do not begin with a full pandas load. Inspect file sizes, compression, schema/header, dtypes, and metadata row counts first; then read approximately 10k–100k rows, fewer for very wide records. Treat early temporal rows as a schema sample, not representative class balance or cardinality. Use additional bounded coverage when needed.

Estimate and label:

- N from reliable metadata or a chunked count; keep estimates distinct from exact counts. Compressed size is not an in-memory size estimate.
- Raw file size and approximate pandas footprint, using sample `memory_usage(deep=True)` per row with dtype/string assumptions and uncertainty. A sample may miss long strings, new categories, or later schema variation.
- Likely encoded D, dense versus sparse output, and memory before each transformation. Account for high-cardinality one-hot expansion, centering/densification, joins, polynomial features, and copies during concatenation.
- Peak working memory, including raw data, encodings, model/solver workspace, CV train/test copies, concurrent workers, and plotting/export buffers; payload size alone is insufficient.
- Whether full in-memory processing is safe, conditional, or unsupported by current evidence. A readable file does not imply a safe full training/sweep workload.

User-provided project machine context: **Apple M5 MacBook Pro, 24 GB unified memory, approximately 858 GB free storage**. Treat these as reported constraints, not live measurements. Check current available RAM, memory pressure/swap, and disk headroom when accessible; otherwise disclose uncertainty. Keep heavy Python workloads comfortably below available memory with room for macOS, other applications, and transient allocations. Unified memory is shared; free storage is not a substitute for RAM. Do not recommend a plan relying heavily on swap. Account for temporary files and repeated experiment outputs when estimating disk needs.

## Chunked profiling

When practical, scan the complete dataset with a default **250,000 rows per chunk**. Reduce this for wide rows or high observed memory; increase only with measured justification. Select required columns and avoid unnecessary DataFrame copies. Aggregate and release chunks; do not retain or concatenate them into the full dataset.

Gather feasible statistics across chunks:

- Total rows, target counts/class balance, and missing-value counts/rates.
- Categorical and identifier cardinalities.
- Numerical and temporal min/max/ranges, temporal coverage, and ordering checks across chunk boundaries.
- Duplicate estimates, repeated-entity/event patterns, and leakage candidates grounded in feature timing and semantics.

Chunking alone does not bound memory if unique-value sets or duplicate trackers grow without limit. Use exact cardinalities/duplicates only when memory-safe; otherwise use bounded estimates, samples, or justified disk-backed processing. Label the method and uncertainty. Do not sum per-chunk distinct counts as global cardinality or treat within-chunk duplicate counts as a global exact result. Include cross-chunk duplicates when claiming exactness; hashes can collide. If a statistic cannot be safely established, report it unknown rather than forcing a full load.

## Transformations and experimental stages

Do not automatically materialize a dense encoded matrix. Estimate first, retain sparse matrices where appropriate, and avoid repeated full-data materialization. For a small profiling pilot, fit transformations only on the appropriate training subset; flag dimensional explosion before proceeding. Sampling cannot establish full-data maximum sparsity or exact encoded cardinality without additional evidence.

Distinguish four stages in recommendations:

1. **Profiling:** header plus 10k–100k rows, followed by bounded chunked statistics where feasible; no model benchmark implied.
2. **Baseline establishment:** suggest about 250k rows for pipeline/debugging, then 500k for a first benchmark, subject to the memory budget and sufficient outcome counts.
3. **Scaling experiments:** suggest 1M rows, then larger justified subsets; measure runtime and peak memory at each step before extrapolating.
4. **Final full-data experiments:** recommend only after preprocessing/solver behavior and memory/runtime costs are understood and feasible.

These scales are examples, not mandatory allocations: reduce for wide data, scarce RAM, or expensive models. Explain rare-class coverage and subset selection. For temporal transactions, inspect ordering and leakage first and use suitable chronological windows/holdouts or combined entity/time splits, never default random CV. Define comparable evaluation protocols across stages and do not reuse the final holdout for tuning.

Start planned measurements with limited concurrency and account for worker multiplication. Before recommending full-scale sweeps, require stable preprocessing dimensionality, a valid leakage-aware split, sufficient class coverage, measured peak memory with headroom, manageable runtime and output storage, and explicit stopping conditions for memory pressure, swapping, or unexpectedly rapid growth. Onboarding recommends these experiments; it does not execute them without a further request.

## Additional report sections

Append these sections to the normal eight-section report:

- **Large-dataset execution plan:** separate profiling, baseline establishment, scaling, and final full-data stages.
- **Estimated memory risks:** file/pandas/encoded/peak estimates, units, machine constraints, expansion risks, uncertainty, and swap avoidance.
- **Safe initial sample size:** proposed rows/columns, rationale, coverage limitations, and rare-class feasibility.
- **Recommended chunk size:** start at 250,000 rows or justify adjustment from sample measurements; explain bounded aggregation.
- **Full-data feasibility:** safe, conditional, or not established, with evidence distinguishing profiling from model training/sweeps.
- **Conditions required before running full-scale sweeps:** measured resource gates, validity checks, storage/runtime budget, and stop criteria.
