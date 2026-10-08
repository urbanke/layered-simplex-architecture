# Initial SCITAS numerical checks — 7 October 2026

Three completed suites passed on the isolated Linux environment: all 263 kernel
cases and 36 special-function cases, all 44 prior/identity checks, and the
2,000-transition Bible chain check at all 55 depths. The largest chain residual
was 9.095e-11 bits; the largest probability residual was 1.380e-13.

The first power suite is preserved with its failed status. All 891 default
components completed; 55 strict-setting components failed their kernel
convergence check, across uniform (19), step (18), Dirichlet(1) (14) and
Dirichlet(1/2) (4). The saved failure diagnostics prompted an independent
high-precision comparison of the integration weights. The correction and its
rerun are recorded separately when complete. Scientific tolerances are unchanged.

The depth suite was still running when this milestone was packaged. Its result
will be recorded separately. These records support the declared finite numerical
cases; combined production admission remains pending.

Each capsule contains an exact complete Run directory, its compute-node launch
record, and its completed scheduler log. The power Run has status `failed` and
is preserved with its complete inventory. Source commit, source files,
interpreter/library versions, settings, input hashes, and individual outcomes
are retained. All four Runs were verified before and after extraction; every
compressed member was also checked against its size and SHA-256. The scientific
source is already preserved in the distributed-validation milestone. The
numerical stores remain identified by the complete preflight hash inventory.
