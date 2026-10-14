# Project policy

- Evaluate Quantum Oracle Sketching scientifically. Negative results are valid; never weaken classical baselines or claim unsupported quantum advantage.
- Distinguish measured performance, simulations, calculated quantities, theoretical resource estimates, and assumptions. Machine-size accounting is not measured quantum hardware usage.
- Preserve validated preprocessing, splitting, metrics, regularization, sketch normalization, RNG order, matrix representation, and resource conventions. Propose methodological changes separately and document deviations from the QOS reference.
- Report ordinary and balanced accuracy for imbalanced classification. Preserve Bank's stratified CV protocol; consider time-aware splits for future temporal datasets.
- Default new seeds to 42; preserve explicitly configured experiment seed lists.
- Preserve sparse matrices where practical, but never silently change a validated representation: Ridge's automatic solver differs between dense and sparse inputs.
- Treat existing datasets and saved results as immutable regression fixtures. Write validation outputs separately; never overwrite fixtures to make tests pass.
- Keep changes within the requested experiment. Do not start another dataset or refactor unrelated notebooks without authorization.
- Extract reusable logic into src only when authorized; keep configuration, figures, and interpretation in notebooks. Establish regression coverage before extraction.
- Record dependency versions and experiment provenance; mark unavailable dates or commits as unknown.

## Known Bank Marketing limitations

- Baseline pipelines fit preprocessing inside each CV fold; sketch sweeps fit it globally before CV. Document this leakage risk; correction requires a separate methodological change.
- Bank accounting uses maximum row sparsity. The inspected QOS reference uses maximum row-or-column sparsity. Preserve and label the Bank convention.
- Bank is a pipeline-validation dataset, not evidence of practical quantum advantage. The QOS formula's +4 represents three logical qubits plus one classical scalar register under the reference accounting.
