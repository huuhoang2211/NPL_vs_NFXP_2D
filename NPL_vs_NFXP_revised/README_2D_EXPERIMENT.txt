2D NPL vs NFXP experiment (build from the experiment in the original codebase: https://github.com/bschjerning/dp_ucph/tree/main/2_dynamic_discrete_choice/zurcher_matlab)
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



Notes
-----
- The experiment remains a two-step comparison: stage 1 estimates
  transition probabilities, and stage 2 compares NPL and NFXP on the same
  estimated transitions.
- NPL timing still includes CCP-initialization time.
- NFXP and NPL initialize the structural parameters at zeros.
