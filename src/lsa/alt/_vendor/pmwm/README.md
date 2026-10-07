# Pinned scientific subset

`provenance.json` records the upstream commit and original/transformed hashes.
The four numerical modules retain upstream mathematics. Only import names and
configuration access are changed. The local kernel shim selects the upstream
Python/NumPy fallback, avoiding implicit native builds and filesystem writes.
The ALT adapter controls settings and uses a read-only store subclass. Importing
these modules directly is not a supported public API. No BPE, memory, corpus,
scheduler, or transformer modules are included. Upstream comments describe its
historical validations, not an ALT accuracy guarantee.
