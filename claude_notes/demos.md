# Demos for xptycho

Each demo reproduces one experiment of the papers with its ground
truth stated.  Full parameters are in
~/claude-notes/xptycho/data-and-verification.md.

1. `demo_1_simulated_known_probe.py`: TCI 2023 Figs. 4(d), 5(d), 6.
   Synthetic ground truth; PMACE with the probe held known.  Expected
   object NRMSE 0.037.
2. `demo_2_simulated_blind_two_mode.py`: TCI 2025 Sec. VI-B.  Blind
   ground truth with two probe modes; the second mode added at
   iteration 20.
3. `demo_3_goldballs_known_probe.py`: TCI 2023 Figs. 10 and 11(d).
   The raw gold-ball scan from CXIDB, preprocessed by xptycho; the
   reference probe held known.
4. `demo_4_goldballs_blind_two_mode.py`: TCI 2025 measured section
   with position refinement.  Parked until 1 to 3 are right.
