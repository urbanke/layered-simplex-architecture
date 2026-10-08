SCITAS scheduling pilot and automatic resource gate

This initial archive contains exact pilot drivers and the independently reviewed resource gate, not completed scientific results. The six production-grid validation pilots preserve original trial IDs and sample grids. Three completed pilots measured uniform primary (20 trials, all 8 n, all 81 depths): 537.22 s / 375.5 MiB peak; depth scaling (20 profiles, all 17 depths): 105.43 s / 303.5 MiB; factorial (100 profiles, d=100000, n=10000, alpha=1.5): 15.59 s / 199.8 MiB. The heavy primary, spectrum and Bible pilots were pending when this immutable snapshot was made.

Late-offset RNG replay passed: spectrum trial_start=980 took 34.43 s (uniform) and 16.67 s (alpha=1.5); factorial trial_start=4500 took 14.67 s. No scientific sampling changes were introduced.

The gate consumes the exact 3904-job plan (block20/factorial500/batch20), complete pilot Runs, corrected power calibration and late-replay records. It verifies hashes, computational source, settings, runtime and single-thread controls, applies 2x timing headroom plus 60 seconds per job and a one-hour controller reserve, then selects the smallest eligible 4/8/16-node count. Each node uses 72 CPUs, 440 GiB and a 24-hour limit. A passed scheduling result does not supply numerical production admission.

The gate output interface is --out DIRECTORY; it creates DIRECTORY/assessment.json. Incomplete inputs produce pending and no node selection. Inputs that fail validation produce failed. Existing output directories are never overwritten.
