# Acquisition instructions and source evidence

Sources inspected 2026-10-16. These links identify upstream sources, not a claim that
an undocumented historical download used a particular URL or version. Nothing here
automatically downloads or overwrites fixtures.

## Bank Marketing

[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
provides `bank.zip`; use `bank-full.csv`, not the additional-data variant. Credit:
Moro, S., Rita, P., & Cortez, P. (2014), Bank Marketing, UCI Machine Learning Repository,
[DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Notebook 02 documents this project's added legacy columns; the committed local CSV
is modified data, not the upstream archive. Preserve attribution and disclose those
modifications. Notebook 05 discards the legacy columns for its validated pipeline.
The original archive hash and acquisition time were not recorded. Use the existing
tracked fixture for exact local reproduction; a fresh upstream file is not assumed
byte-identical to that fixture.

## PaySim

[Provider's Kaggle page](https://www.kaggle.com/datasets/ealaxi/paysim1)
is the candidate acquisition source for the synthetic transaction data. Download
manually under the provider's current access/terms, preserve the downloaded filename
and version, and compare with the registered `data/paysim.csv` hash before use.
The inspected page did not expose verifiable license text; no license is inferred
from third-party summaries or the simulator code. Exact historical download and
rename provenance are unknown. Keep bytes local; no encoding/model run is authorized
by these instructions.

## CFPB

The [official narratives archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
links “CCDB Export January 2026 through February 2026.” Obtain that archive, extract
without overwriting the frozen file, and compare the extracted CSV hash with
`cfpb_raw_2026_01_02`. The original local ZIP is absent; no ZIP hash is claimed.
Do not substitute the live database or August export: exports can differ in
coverage, labels, and narrative availability. A new download is acceptable for exact
reproduction only if the extracted hash matches.

The [CFPB data page](https://www.consumerfinance.gov/data-research/consumer-complaints/)
states public complaint data may be used, analyzed, and built on. This project still
keeps raw complaint text and vocabulary tokens local; no broader license or privacy
clearance is inferred. The unused August file's historical acquisition is also unknown.

## PBMC68k reference

The local reference commit is `a3fe8d6fe777750749f5e90131180e016088bdaf`.
`real_datasets/pbmc68k_utils.py` calls `scvelo.datasets.pbmc68k` with the
CWD-relative cache `./data_cache/pbmc68k.h5ad`; consult
[scVelo's dataset documentation](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.pbmc68k.html)
for the Zheng et al. 2017 source. Use a separate reference environment if reproducing
that benchmark; do not install/upgrade the project stack merely to fetch it.
No PBMC file was found locally and no exact size, hash, filtered N/D or sparsity is
claimed. Verify dataset-specific redistribution terms before any publication; the
reference repository's code license does not settle data licensing.
