2D NPL vs NFXP experiment (revised MATLAB bundle)
=======================================================

Files
-----
- run_mc_npl_nfxp_2d.m : main Monte Carlo driver
- zurcher2d.m          : 2D model primitives, Bellman operators, likelihood, simulator
- npl2d.m              : 2D NPL estimator revised to:
                         (i) precompute Finv = inv(I-beta*Fu) once per outer iteration,
                         (ii) pass that inverse into the inner pseudo-likelihood,
                         (iii) avoid repeated utility evaluation inside ll/phi/lambda
- nfxp2d.m             : 2D NFXP estimator
- dpsolver.m           : fixed-point solver copied from the original codebase
- struct2vec.m         : helper copied from the original codebase
- vec2struct.m         : helper copied from the original codebase

How to run
----------
1. Put all files from this folder in one MATLAB directory.
2. Open MATLAB in that directory.
3. Run:
   results = run_mc_npl_nfxp_2d();

What changed relative to the uploaded repository
------------------------------------------------
1. The NPL estimator now follows the original 1D spirit more closely:
   - For each outer iteration K, it precomputes
         Finv = inv(I - beta * Fu(pk_{K-1}))
     once and passes Finv into the inner likelihood.
   - The inner likelihood computes u only once and passes u into phi() and
     lambda(), removing duplicated utility evaluation.

2. The initial CCPs for NPL now use a direct 2D frequency extension of
   original case 4 instead of the smoothed frequency initializer:
   - On occupied states:
         pK(s) = (# keep decisions observed at state s) / (# visits to state s)
   - On empty states:
         pK(s) is filled with the sample-wide keep frequency so the
         initializer remains well-defined in sparse 2D state spaces.
   - The resulting CCPs are clipped to [1e-6, 1-1e-6].

Notes
-----
- The experiment remains a two-step comparison: stage 1 estimates
  transition probabilities, and stage 2 compares NPL and NFXP on the same
  estimated transitions.
- NPL timing still includes CCP-initialization time.
- NFXP still starts from zeros.
