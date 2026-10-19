# Publication sanitation and retained phase artifacts

Date: 2026-10-19. This is publication metadata, not a new experiment.

The six local-checkout prefixes in `kappa4_cap/failure.txt` were removed,
leaving repository-relative source paths. Traceback line numbers, source
statements, the exception, and all numerical values are unchanged.
Only that text artifact's existing integrity entry is replaced in
`gate_manifest.json`; this note receives a new integrity entry.

- Original traceback SHA-256: `550837cca89b78c1ed01f54e9ff25cd3a21706e854e4d21b067dabbb3515a6ac`
- Sanitized traceback SHA-256: `43cdecb31214339ba4ba45715803b30af0425e230a0186fa8ef4195274b2e529`

The original optimization commit was `a480ebc`. It and the other three
unpublished commits were reconstructed with their original boundaries and
messages. Historical run/source provenance continues to identify the original
execution revisions; this sanitation does not pretend the runs were repeated
under the replacement commits. The first two commit trees and the streaming
plan are unchanged. Published history is unchanged.

## Intentional scientific phase artifacts

The five NPZ files under `phase_cache/` are retained reproducibility artifacts
for the phase-synthesis optimization and cache-reuse measurements. They are
not raw datasets, random oracle realizations, or disposable application caches.
Each contains only float64 `coefficients` and `angles` arrays and a Unicode
JSON `metadata` scalar with numerical tolerances, convergence/timing records,
algorithm/dependency versions, reference/code identities, and integrity hashes.
They contain no pickled objects, machine-specific paths, or credentials.

| Cache identity prefix | Coefficients / angles each | File bytes |
| --- | ---: | ---: |
| `04ab07371254` | 50 | 4,882 |
| `33b66709dd76` | 182 | 7,002 |
| `49c34910f33b` | 4,536 | 76,658 |
| `7ba048c4031c` | 502 | 12,118 |
| `f0007580d75c` | 2,266 | 40,338 |
| **Total** | | **140,998** |

The publication audit loads every archive with `allow_pickle=False`, checks
finite numerical arrays and shape agreement, verifies the coefficient and
combined-array SHA-256 values, reconstructs each cache identity/filename,
and verifies the recorded convergence residual against its recorded tolerance.
All original archive file hashes match `gate_manifest.json`; all five files
are preserved byte-for-byte.

The original experiment's verified-angle path checks the reference circuit
response on 2,001 nodes with tolerance 1e-10 on both cache creation and reuse.
Those historical numerical checks and the independent circuit diagnostics are
reported in [the optimization report](../../PHASE_B_OPTIMIZATION.md).
This publication audit verifies their archived integrity; it does not rerun
high-degree synthesis or claim a new uniform approximation proof. Reusable
Clifford readout and complete theorem-level sampled-oracle composition remain
unimplemented, as documented in the fidelity audit.
