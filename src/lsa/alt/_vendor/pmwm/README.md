# Pinned scientific subset

`provenance.json` records the upstream commit, original/transformed hashes,
and every local change. Imports and configuration access are isolated. The
local kernel shim selects the upstream Python/NumPy fallback, avoiding implicit
native builds and source-directory writes.

ALT calibration found large low-count errors in the old high-depth saddle and
unchecked small-t-series provider. The high-depth route now uses direct contour
integration with a checked 1e-13 series threshold. Its legacy private name and
configuration cutoff are retained for compatibility; diagnostics identify the
actual direct-contour provider. The original saddle approximation remains only
for explicit validation comparisons. Contour oversampling now controls the
spacing, and its tail target is explicit. See the declared kernel-calibration
cases and records before drawing conclusions about a numerical domain.

The ALT adapter uses a read-only store subclass. Direct module imports are not
the supported experiment API. No BPE, memory, corpus, scheduler, or transformer
modules are included. Upstream historical comments are not ALT guarantees.
