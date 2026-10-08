# Corpus used for ALT preparation

`kjv.txt.gz` is the repository's pinned King James Bible token stream, derived
from Project Gutenberg eBook #10. The preprocessing recipe is documented in
`scripts/get_kjv.py`: retain the Gutenberg body, remove verse numbers, handle
curly-apostrophe possessives, remove separator characters, and tokenize words
and punctuation while preserving case.

`kjv.manifest.json` records the compressed and decompressed byte hashes, the
whitespace-separated token count, distinct-type count, and prefix statistics.
The verified bundled stream has 915,860 tokens and 13,550 types. The historical
Overleaf archive includes other tokenizations; keep their dataset identities
separate during comparison.

`python scripts/get_kjv.py` reuses an existing `data/kjv.txt`, or extracts the
bundled stream offline, and checks token and prefix counts. Before a production
run, also verify the decompressed byte hash against `kjv.manifest.json`. The
extracted text and downloaded source remain local and ignored by Git.
Production run records reference the pinned corpus hash and the exact
preprocessing/source revision. A future change of corpus is a protocol revision.

## Retained second-tokenization control

`kjv-secondary.txt.gz` contains the current appendix control's 917868 tokens and
13554 types. It was deterministically recovered from historical Overleaf
`data/kjv_clean.txt` by Python Unicode regex `\w+|[^\w\s]`, preserving case.
`kjv-secondary.manifest.json` records the original archive/member hashes, exact
recipe, and compressed/decompressed hashes. It is a distinct dataset from the
canonical corpus.
