SCITAS powered-model strict-refinement diagnosis, 2026-10-07

Original campaign: frozen source cc1524a636a5f1996b4fbff1668c477ed0e989b8, job 40152958.
All 891 default components passed. Of 891 strict components, 55 failed: uniform 19, step 18, Dirichlet(1) 14 and Dirichlet(1/2) 4. Every failure is the r=0 kernel; all failure records remain unchanged.

The two compute-node probes 40157333 and 40158239 reproduce the failure and compare independent rules. SciPy 1.18 reuses the Legendre derivative from before its Newton node correction in its weights. The quadrature nodes agree across platforms within 2.22e-16 but the 256-node weights differ by up to 3.79e-14 absolute / 3.36e-10 relative. Transplanting Linux rules into the Mac evaluation reproduces Linux kernels to 2.22e-16, isolating quadrature rules from kernel arithmetic.

For r=0,w=59,v=0.02737464341427795, the 80-digit direct E-coordinate reference is -0.4484912791112434003184478741673991552290 nats. Original Linux rules produce errors +2.648992136755623e-13 (256 nodes) and -3.9623859748871837e-13 (512 nodes), yielding the rejected 6.611378111642807e-13 refinement difference.

Repair commit f0a5ebd0a57d46caeb9bf45407c062f37b8d64bc recomputes Legendre weights using the derivative evaluated at the returned nodes. This is a numerical rule correction, with all scientific tolerance and model settings unchanged. Across five selected kernels, corrected 512-node errors are at most 1.20e-14 on Linux and 1.11e-14 on Mac, against 80-digit independent E-domain integrals. Independent 80-digit Newton/Legendre recurrence rules further confirm the diagnosis.

All 55 exact originally failed kernels now converge locally under their original strict settings and agree with independent 60-digit E-domain integrals to 1.6883684275850313e-14 nats. These kernel checks are diagnosis evidence, not a replacement for the full saved-profile calibrations. The new targeted regression tests pass (37), and the full appendix script passes. The initial full regression log preserves two sandbox shared-memory failures; the final unrestricted regression will be recorded separately after admission-builder changes settle. Full local and Linux power recalibrations run in new output directories.
