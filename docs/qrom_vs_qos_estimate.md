# QROM vs QOS: per-query loading cost for the CFPB training matrix

Date: 2026-10-20. **Label: back-of-envelope resource estimate.**
Generator: `scripts/qrom_vs_qos.py`. Output: [`results/audit/qrom_vs_qos/`](../results/audit/qrom_vs_qos/) (`qrom_costs.csv`, plus `summary.json` with input and output hashes). Formula tests: `tests/test_qrom_formulas.py`.

## Headline finding

**Every single block-encoding query costs QOS about 4×10¹⁸ passes over the CFPB training matrix. A finite dataset therefore has to be stored and re-streamed anyway, so the storage QOS is meant to save is not saved.**

- **The QOS cost.** About 6×10²⁴ samples per A/2 query, at the measured exponent s^3.15 and ε = 10⁻². Divided by the 1,563,584 nonzeros, that is **≈ 4×10¹⁸ re-reads of every nonzero, per query**, before multiplying by the solver's query count.
- **Why storage isn't avoided.** No finite data source can be sampled that many times except by replaying stored data. The bank keeps the full dataset either way. Producing frozen-TF-IDF samples also requires the 8.2 MB featurizer state measured in the classical streaming study.
- **The QROM cost.** QROM stores the same table (9.4 MB CSR; 12.8 MB with the CSC column-access copy) and reads it **once** per query, at 10⁵–10⁶ Toffoli per A/s query in practical dirty-qubit regimes.
- **So:** QOS saves the ~9–13 MB stored table only inside the quantum machine, at the cost of ~10²⁴–10²⁵ samples per query drawn from a source that must itself be stored.

## The matrix

From audit step 1 (the full January training matrix, as used by the one-vs-rest tasks):

| Quantity | Value |
|---|---|
| N rows | 13,459 |
| D columns | 50,000 |
| nnz | 1,563,584 |
| s_row / s_col (s_row_or_col) | 1,254 / 8,776 |
| Bits per entry | value 32 (float32); column index ⌈log₂ D⌉ = 16; row index ⌈log₂ N⌉ = 14 |

One query to the A/s block encoding (Lemma D.22) uses four lookups: the row-index oracle, the column-index oracle, and the element oracle **twice** (O_ele and O_ele†). Each oracle is costed as one QROM lookup.

## QROM formulas (exact, as printed)

L = table entries, b = bits per entry, λ = power-of-two block size.

| Construction | Toffoli count | Dirty qubits | Source |
|---|---|---|---|
| Unary iteration | ≈ L | 0 | Babbush et al. 2018, as quoted in arXiv:2605.20334 |
| SelectSwap (dirty) | 2L/λ + 4b(λ−1) | b(λ−1) | Berry et al. 2019, as quoted in arXiv:2605.20334 |
| SelectCopy | 2L/λ + 2b(λ−1) + 2λ − 6 | b(λ−1) | arXiv:2605.20334 §II.B |
| Sequential bit packets | (⌈b/μ⌉+1)(⌈L/λ⌉+λ−3) + (λ−1)(μ(⌊b/μ⌋+1) + b mod μ); α = b gives (1+1/b)L/λ + 2(b+1)(bλ−2) | μ(λ−1) | arXiv:2605.20334 Thm 1 (Eq. A1), Eq. 7 |
| Dense encoding | L/λ + (4b+2)λ − 6 | bλ | arXiv:2610.02321 Eq. 6 |
| Dense + sequential packets | (m+1)(Q+λ−3) + 2λ(b+μ), with m = ⌈b/μ⌉ and Q = ⌈L/(2λ)⌉; α = b gives ½(1+1/b)L/λ + 3b(b+1)λ − 3(b+1) | μλ | arXiv:2610.02321 Eq. 9, Eq. 8 |

**How the formulas were checked.** In the qubit-constrained regime the leading terms are 2L/λ (SelectSwap), (1+1/b)L/λ (sequential packets), L/λ (dense) and ½(1+1/b)L/λ (combined). So the combined construction is about 3.9× better than SelectSwap at b = 32, as both papers state. The tests check these leading-term ratios, and also that the general bounds reduce exactly to SelectCopy and dense encoding at μ = b.

**How the table was computed.**
- λ and μ are minimized over powers of two (λ) and 1 ≤ μ ≤ b.
- Dirty-qubit budgets are 256 and 1,024 (the papers' figure settings), plus an unconstrained case.
- Dirty qubits can be any idle qubits, as long as their states are restored. Beyond them, each construction needs an address register of ⌈log₂ L⌉ qubits (21 packed, 29 padded), b output qubits, and at most about log₂ L clean helpers.

## QROM cost per query

**Packed tables** — L = nnz = 1,563,584 per oracle (lower bound; ignores CSR/CSC offset arithmetic).

| Construction | Toffoli per A/s query: 256 / 1,024 / unconstrained dirty qubits | Dirty qubits used (unconstrained) | Toffoli per A/2 query (× D.23 amplification, unconstrained) |
|---|--:|--:|--:|
| Unary iteration | 6.3e+06 / 6.3e+06 / 6.3e+06 | 0 | 1.6e+12 |
| SelectSwap (dirty) | 1.2e+06 / 3.1e+05 / 1.4e+05 | 4,080 | 3.4e+10 |
| SelectCopy | 1.2e+06 / 3.0e+05 / 9.9e+04 | 8,160 | 2.5e+10 |
| Sequential bit packets (opt. μ) | 6.4e+05 / 2.1e+05 / 9.9e+04 | 8,160 | 2.5e+10 |
| Dense encoding | 5.9e+05 / 1.6e+05 / 9.8e+04 | 4,096 | 2.5e+10 |
| Dense + sequential (opt. μ) | 3.4e+05 / 1.3e+05 / 9.8e+04 | 4,096 | 2.5e+10 |

**Padded tables** — row/element L = N·s_row = 16,877,586; column L = D·s_col = 438,800,000.

| Construction | Toffoli per A/s query: 256 / 1,024 / unconstrained dirty qubits | Dirty qubits used (unconstrained) | Toffoli per A/2 query (× D.23 amplification, unconstrained) |
|---|--:|--:|--:|
| Unary iteration | 4.9e+08 / 4.9e+08 / 4.9e+08 | 0 | 1.2e+14 |
| SelectSwap (dirty) | 6.5e+07 / 1.6e+07 / 8.0e+05 | 57,330 | 2.0e+11 |
| SelectCopy | 6.5e+07 / 1.6e+07 / 6.0e+05 | 57,330 | 1.5e+11 |
| Sequential bit packets (opt. μ) | 3.1e+07 / 7.9e+06 / 6.0e+05 | 57,330 | 1.5e+11 |
| Dense encoding | 3.3e+07 / 8.2e+06 / 6.0e+05 | 28,672 | 1.5e+11 |
| Dense + sequential (opt. μ) | 1.6e+07 / 4.1e+06 / 6.0e+05 | 28,672 | 1.5e+11 |

**The A/s → A/2 amplification is counted for both methods.** The 1/s subnormalization comes from the sparse-oracle block-encoding construction (Lemma D.22 / Lemma 48: one query each to the row-index, column-index and element oracles yields A/s). That holds whether those oracles are QROM lookups or QOS sketches. Any solver built on this construction pays the factor s somewhere: either as Theorem D.23's degree-O(s) amplification to A/2, or as an s-times-larger effective condition number in its own polynomial degree. So the A/2 column multiplies the A/s cost by the same amplification degree for both methods: my measured D.23 polynomial degree, about 28.7·s ≈ 2.5×10⁵ at s = 8,776. Because the factor is identical, it cancels in the comparison. The A/s and A/2 columns therefore give the same QROM-to-QOS ratio, and the A/2 column matches the unit of the QOS per-query figure. A QROM loader could instead feed a different block-encoding family (for example one with Frobenius-norm subnormalization). That would be a different construction, outside this comparison.

## QOS cost per query

From [docs/empirical_f61.md](empirical_f61.md) (one-vs-rest matrix, s = 8,776, padded dimension 65,536):

| Exponent used for the block-encoding s⁵ | Samples per A/2 query, ε = 10⁻² | Samples per A/2 query, ε = 10⁻³ |
|---|--:|--:|
| Measured, N = 64 fit (s^3.15) | 6.2×10²⁴ | 6.3×10²⁵ |
| Measured, N = 128 fit (s^3.27) | 1.3×10²⁵ | 1.3×10²⁶ |
| Proof s⁵, anchored to the same measured point | 2.0×10²⁹ | 2.1×10³⁰ |

**Qubits.** The reference's machine-size accounting (A9) gives 52 logical qubits for this matrix. The theorem's F60 polynomial prefactor is far larger (audit: 10¹²–10¹⁸ without b). QOS holds no data table in the quantum machine.

## Side by side (per query, practical regime: 1,024 dirty qubits, packed tables)

| | QROM (dense + sequential packets) | QOS (empirical F61) |
|---|---|---|
| Unit | Toffoli gates | data samples |
| Per A/s query | 1.3×10⁵ | not separately computed |
| Per A/2 query | 3.3×10¹⁰ (with D.23 amplification, counted for both: it is a property of the sparse block encoding) | 6×10²⁴ – 1×10²⁵ |
| Quantum memory | about 1,080 qubits: 1,024 borrowed dirty, 21 address, up to 32 output, ≤ 21 clean helpers | about 52 logical qubits (A9 accounting) |
| Must the data be stored? | Yes: the classical controller holds the 9–13 MB table and reads all of it once per query, as Clifford-specified loads | Not in the machine. In practice the source must be re-streamed about 4×10¹⁸ times per query, plus the 8.2 MB featurizer |

## Caveats (all carried forward)

- **The units differ.** Toffoli gates and samples aren't interchangeable. This compares orders of magnitude only, not a cost ratio.
- **Per query only.** Neither column includes the solver's query count Q (F97), which multiplies both.
- **The QOS figure is extrapolated** from a simulation at N ≤ 128 and s ≤ 32, with uniform sparsity, under the expected-unitary model, by 2–3 decades in s. **Its padded-dimension factor, linear in size, is the theorem's, not a measurement.**
- **The packed QROM figures are lower bounds.** They ignore CSR/CSC offset arithmetic. The padded figures are an upper reference.
- **Formula assumptions.** The QROM bounds count Toffoli at the logical level, assuming measurement-based uncomputation. They exclude Clifford cost, routing, latency and fault-tolerance overhead. Bit widths assume float32 values; quantizing would shrink b.
