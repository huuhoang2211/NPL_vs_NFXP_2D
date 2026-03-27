function results = run_mc_npl_nfxp_2d()
% RUN_MC_NPL_NFXP_2D
% Monte Carlo comparison of NPL and NFXP in a
% two-dimensional dynamic discrete choice model.
%
% The experiment is designed to stay close to the original Rust/Zurcher
% MATLAB architecture:
%   1) A model class (zurcher2d.m) holds utilities, transitions, Bellman
%      operators, likelihood, simulator, and equilibrium distribution.
%   2) NPL (npl2d.m) keeps the nested pseudo-likelihood structure with an
%      outer CCP iteration and an inner trust-region optimization.
%      The inner problem precomputes Finv once per outer iteration, and the
%      initializer is a direct 2D frequency extension of original case 4.
%   3) NFXP (nfxp2d.m) keeps the nested fixed-point logic and solves the
%      Bellman equation at every structural parameter evaluation.
%
% Design choices for a fair comparison:
%   * Both estimators use exactly the same simulated sample in each draw.
%   * Both estimators use exactly the same first-stage transition estimates.
%   * First-stage transition estimation time is excluded from both runtimes,
%     because it is common to both estimators.
%   * NPL runtime includes the CCP-initialization time, because that is part
%     of the NPL estimator.
%   * NFXP starts from zeros, not from the NPL estimate.
%
% Usage:
%   results = run_mc_npl_nfxp_2d();

    clear global zurcher2d_V0 npl2d_pk_last
    close all; clc;

    settings = default_settings();

    % --------------------------
    % DGP: 2D extension of Rust
    % --------------------------
    mp0 = struct();
    mp0.n1 = 50;                        % mileage-like dimension
    mp0.n2 = 8;                         % slow wear/condition dimension
    mp0.p1 = [0.0937 0.4475 0.4459 0.0127]';   % original Rust-style support 0:4
    mp0.p2 = 0.85;                      % support 0:1 for slow dimension
    mp0.RC = 10.80;
    mp0.c1 = 120.0;
    mp0.c2 = 650.0;
    mp0.beta = 0.995;                   % slightly below Rust default for 2D numerical stability
    mp0.bellman_type = 'iv';
    mp0.pnames_u = {'RC','c1','c2'};
    mp0.pnames_P = {};                  % keep two-step partial MLE in stage 2
    mp_true = zurcher2d.setup(mp0);

    fprintf('============================================================\n');
    fprintf('2D Monte Carlo comparison: NPL vs NFXP\n');
    fprintf('============================================================\n');
    fprintf('State space: n1=%d, n2=%d, total states=%d\n', mp_true.n1, mp_true.n2, mp_true.n);
    fprintf('Monte Carlo draws: %d\n', settings.nMC);
    fprintf('Sample size per draw: N=%d, T=%d, NT=%d\n', settings.N, settings.T, settings.N*settings.T);
    fprintf('Bellman representation: %s\n', mp_true.bellman_type);
    fprintf('Structural parameters: RC=%g, c1=%g, c2=%g, beta=%g\n', ...
        mp_true.RC, mp_true.c1, mp_true.c2, mp_true.beta);
    fprintf('------------------------------------------------------------\n');

    % True model solution and stationary initial distribution
    P_true = zurcher2d.statetransition(mp_true);
    [V_true, pk_true] = zurcher2d.solve_model(mp_true, P_true);
    pp_true = zurcher2d.eqb(mp_true, P_true, pk_true);

    % Preallocate results
    k = numel(struct2vec(mp_true, mp_true.pnames_u));
    results = initialize_results(settings, mp_true, k);
    representative_saved = false;

    for iMC = 1:settings.nMC
        seed = settings.seed0 + iMC - 1;

        % --------------------------
        % Simulate one Monte Carlo draw
        % --------------------------
        data = zurcher2d.simdata(settings.N, settings.T, mp_true, P_true, pk_true, seed, pp_true);

        % Shared first-stage transition estimation (same for both estimators)
        t_stage1 = tic;
        mp_stage1 = zurcher2d.estimate_transitions(data, mp_true);
        P_hat = zurcher2d.statetransition(mp_stage1);
        stage1_time = toc(t_stage1);

        % Initial CCP for NPL (direct 2D extension of original case 4;
        % counts as part of NPL runtime)
        t_ccp = tic;
        [pk_init, init_info] = zurcher2d.ccp_init_nonparam(data, mp_stage1, settings.init);
        ccp_init_time = toc(t_ccp);

        theta0 = zeros(k,1);

        % --------------------------
        % NPL
        % --------------------------
        mp_npl0 = mp_stage1;
        mp_npl0 = zurcher2d.set_theta(mp_npl0, theta0);

        npl_error = '';
        try
            [res_npl, det_npl] = npl2d.estim(theta0, pk_init, data, P_hat, mp_npl0, ...
                settings.Kmax, settings.npl_tol_theta, settings.npl_tol_pk, settings.verbose_estimators);
            res_npl.cputime_total = res_npl.cputime + ccp_init_time;
            res_npl.policy_rmse = sqrt(mean((det_npl.pk_fix - pk_true).^2));
            det_npl.init_info = init_info;
            det_npl.init_ccp = pk_init;
        catch ME
            res_npl = nan_result_struct(mp_true.pnames_u);
            res_npl.converged = false;
            res_npl.cputime = NaN;
            res_npl.cputime_total = NaN;
            res_npl.outerIter = NaN;
            res_npl.MajorIter = NaN;
            res_npl.funcCount = NaN;
            res_npl.llval = NaN;
            res_npl.policy_rmse = NaN;
            det_npl = struct();
            npl_error = ME.message;
        end

        % --------------------------
        % NFXP
        % --------------------------
        mp_nfxp0 = mp_stage1;
        mp_nfxp0 = zurcher2d.set_theta(mp_nfxp0, theta0);  % starts from zeros by construction

        nfxp_error = '';
        try
            [res_nfxp, det_nfxp] = nfxp2d.estim(data, mp_nfxp0, theta0, settings.verbose_estimators);
            res_nfxp.policy_rmse = sqrt(mean((det_nfxp.pk_hat - pk_true).^2));
        catch ME
            res_nfxp = nan_result_struct(mp_true.pnames_u);
            res_nfxp.converged = false;
            res_nfxp.cputime = NaN;
            res_nfxp.MajorIter = NaN;
            res_nfxp.funcCount = NaN;
            res_nfxp.llval = NaN;
            res_nfxp.policy_rmse = NaN;
            det_nfxp = struct();
            nfxp_error = ME.message;
        end

        % --------------------------
        % Store Monte Carlo outputs
        % --------------------------
        results = store_draw_results(results, iMC, mp_stage1, pk_true, pk_init, ...
            init_info.occupied_share, ccp_init_time, stage1_time, res_npl, res_nfxp);

        if ~isempty(npl_error)
            results.npl.error_messages{iMC} = npl_error;
        end
        if ~isempty(nfxp_error)
            results.nfxp.error_messages{iMC} = nfxp_error;
        end

        % Save one representative successful draw for plotting
        if settings.do_plots && ~representative_saved && res_npl.converged && res_nfxp.converged
            results.representative = make_representative_struct(mp_true, pk_true, V_true, ...
                pk_init, init_info, mp_stage1, det_npl, det_nfxp);
            representative_saved = true;
        end

        if mod(iMC, settings.print_every)==0 || iMC==1 || iMC==settings.nMC
            fprintf(['Draw %3d/%3d | occ.=%.3f | initRMSE=%.4f | ' ...
                     'NPL conv=%d time=%.3f s (K=%g) | ' ...
                     'NFXP conv=%d time=%.3f s | stage1=%.3f s\n'], ...
                    iMC, settings.nMC, init_info.occupied_share, ...
                    sqrt(mean((pk_init-pk_true).^2)), ...
                    res_npl.converged, res_npl.cputime_total, res_npl.outerIter, ...
                    res_nfxp.converged, res_nfxp.cputime, stage1_time);
        end
    end

    % Summaries
    results.summary = summarize_results(results, mp_true);

    fprintf('\n============================================================\n');
    fprintf('Monte Carlo summary\n');
    fprintf('============================================================\n');
    disp(results.summary.parameter_table);
    fprintf('\n');
    disp(results.summary.performance_table);

    fprintf('Average CCP-initializer RMSE   : %.6f\n', results.summary.avg_initializer_rmse);
    fprintf('Average CCP-initializer time   : %.6f s\n', results.summary.avg_initializer_time);
    fprintf('Average occupied-state share   : %.6f\n', results.summary.avg_occupied_share);

    if settings.save_results
        save(settings.results_file, 'results');
        fprintf('\nSaved results to %s\n', settings.results_file);
    end

    if settings.do_plots && representative_saved
        plot_representative_surfaces(results.representative);
    end
end

% ======================================================================
% Local helpers
% ======================================================================

function settings = default_settings()
    settings = struct();
    settings.nMC = 100;
    settings.N = 300;
    settings.T = 30;
    settings.seed0 = 20260327;
    settings.Kmax = 20;
    settings.npl_tol_theta = 1.0e-6;
    settings.npl_tol_pk = 1.0e-8;
    settings.print_every = 5;
    settings.verbose_estimators = false;
    settings.do_plots = true;
    settings.save_results = true;
    settings.results_file = 'results_mc_npl_nfxp_2d.mat';

    settings.init = struct();
    settings.init.method = 'frequency';
    settings.init.clip = 1e-6;
    settings.init.empty_fill = 'overall_keep';
    settings.init.empty_value = [];
end

function results = initialize_results(settings, mp_true, k)
    nMC = settings.nMC;
    results = struct();
    results.settings = settings;
    results.true_mp = mp_true;
    results.theta_true = struct2vec(mp_true, mp_true.pnames_u);

    results.stage1.p1_hat = nan(numel(mp_true.p1), nMC);
    results.stage1.p2_hat = nan(numel(mp_true.p2), nMC);
    results.stage1.time = nan(1, nMC);

    results.initializer.ccp_rmse = nan(1, nMC);
    results.initializer.time = nan(1, nMC);
    results.initializer.occupied_share = nan(1, nMC);

    results.npl.theta = nan(k, nMC);
    results.npl.cputime = nan(1, nMC);
    results.npl.cputime_total = nan(1, nMC);
    results.npl.outerIter = nan(1, nMC);
    results.npl.MajorIter = nan(1, nMC);
    results.npl.funcCount = nan(1, nMC);
    results.npl.llval = nan(1, nMC);
    results.npl.converged = false(1, nMC);
    results.npl.policy_rmse = nan(1, nMC);
    results.npl.error_messages = cell(1, nMC);

    results.nfxp.theta = nan(k, nMC);
    results.nfxp.cputime = nan(1, nMC);
    results.nfxp.MajorIter = nan(1, nMC);
    results.nfxp.funcCount = nan(1, nMC);
    results.nfxp.llval = nan(1, nMC);
    results.nfxp.converged = false(1, nMC);
    results.nfxp.policy_rmse = nan(1, nMC);
    results.nfxp.error_messages = cell(1, nMC);

    results.representative = struct();
    results.summary = struct();
end

function results = store_draw_results(results, iMC, mp_stage1, pk_true, pk_init, ...
                                      occupied_share, ccp_init_time, stage1_time, res_npl, res_nfxp)
    theta_npl = pack_theta(res_npl, results.true_mp.pnames_u);
    theta_nfxp = pack_theta(res_nfxp, results.true_mp.pnames_u);

    results.stage1.p1_hat(:,iMC) = mp_stage1.p1(:);
    results.stage1.p2_hat(:,iMC) = mp_stage1.p2(:);
    results.stage1.time(iMC) = stage1_time;

    results.initializer.ccp_rmse(iMC) = sqrt(mean((pk_init - pk_true).^2));
    results.initializer.time(iMC) = ccp_init_time;
    results.initializer.occupied_share(iMC) = occupied_share;

    results.npl.theta(:,iMC) = theta_npl;
    results.npl.cputime(iMC) = res_npl.cputime;
    results.npl.cputime_total(iMC) = res_npl.cputime_total;
    results.npl.outerIter(iMC) = res_npl.outerIter;
    results.npl.MajorIter(iMC) = res_npl.MajorIter;
    results.npl.funcCount(iMC) = res_npl.funcCount;
    results.npl.llval(iMC) = res_npl.llval;
    results.npl.converged(iMC) = logical(res_npl.converged);
    results.npl.policy_rmse(iMC) = res_npl.policy_rmse;

    results.nfxp.theta(:,iMC) = theta_nfxp;
    results.nfxp.cputime(iMC) = res_nfxp.cputime;
    results.nfxp.MajorIter(iMC) = res_nfxp.MajorIter;
    results.nfxp.funcCount(iMC) = res_nfxp.funcCount;
    results.nfxp.llval(iMC) = res_nfxp.llval;
    results.nfxp.converged(iMC) = logical(res_nfxp.converged);
    results.nfxp.policy_rmse(iMC) = res_nfxp.policy_rmse;
end

function rep = make_representative_struct(mp_true, pk_true, V_true, pk_init, init_info, mp_stage1, det_npl, det_nfxp)
    rep = struct();
    rep.mp_true = mp_true;
    rep.pk_true = pk_true;
    rep.V_true = V_true;
    rep.pk_init = pk_init;
    rep.init_info = init_info;
    rep.mp_stage1 = mp_stage1;
    rep.mp_npl = det_npl.mp_hat;
    rep.mp_nfxp = det_nfxp.mp_hat;
    rep.pk_npl = det_npl.pk_fix;
    rep.pk_nfxp = det_nfxp.pk_hat;
end

function theta = pack_theta(res, pnames)
    theta = nan(numel(pnames),1);
    for i = 1:numel(pnames)
        if isfield(res, pnames{i})
            theta(i) = res.(pnames{i});
        end
    end
end

function res = nan_result_struct(pnames)
    res = struct();
    for i = 1:numel(pnames)
        res.(pnames{i}) = NaN;
    end
    res.cputime = NaN;
    res.cputime_total = NaN;
    res.outerIter = NaN;
    res.MajorIter = NaN;
    res.funcCount = NaN;
    res.llval = NaN;
    res.converged = false;
    res.policy_rmse = NaN;
end

function summary = summarize_results(results, mp_true)
    pnames = mp_true.pnames_u;
    theta_true = struct2vec(mp_true, pnames);

    [mean_npl, sd_npl, rmse_npl, bias_npl] = summarize_theta(results.npl.theta, theta_true, results.npl.converged);
    [mean_nfxp, sd_nfxp, rmse_nfxp, bias_nfxp] = summarize_theta(results.nfxp.theta, theta_true, results.nfxp.converged);

    param_name = pnames(:);
    summary.parameter_table = table( ...
        param_name, theta_true, ...
        mean_npl, bias_npl, sd_npl, rmse_npl, ...
        mean_nfxp, bias_nfxp, sd_nfxp, rmse_nfxp, ...
        'VariableNames', {'Parameter','TrueValue', ...
                          'Mean_NPL','Bias_NPL','SD_NPL','RMSE_NPL', ...
                          'Mean_NFXP','Bias_NFXP','SD_NFXP','RMSE_NFXP'});

    conv_npl = results.npl.converged;
    conv_nfxp = results.nfxp.converged;

    method = {'NPL'; 'NFXP'};
    converged_runs = [sum(conv_npl); sum(conv_nfxp)];
    convergence_rate = [mean(conv_npl); mean(conv_nfxp)];
    avg_cpu = [mean_or_nan(results.npl.cputime_total, conv_npl); ...
               mean_or_nan(results.nfxp.cputime, conv_nfxp)];
    med_cpu = [median_or_nan(results.npl.cputime_total, conv_npl); ...
               median_or_nan(results.nfxp.cputime, conv_nfxp)];
    avg_outer = [mean_or_nan(results.npl.outerIter, conv_npl); NaN];
    avg_major = [mean_or_nan(results.npl.MajorIter, conv_npl); mean_or_nan(results.nfxp.MajorIter, conv_nfxp)];
    avg_func = [mean_or_nan(results.npl.funcCount, conv_npl); mean_or_nan(results.nfxp.funcCount, conv_nfxp)];
    avg_logl = [mean_or_nan(results.npl.llval, conv_npl); mean_or_nan(results.nfxp.llval, conv_nfxp)];
    avg_policy_rmse = [mean_or_nan(results.npl.policy_rmse, conv_npl); mean_or_nan(results.nfxp.policy_rmse, conv_nfxp)];

    summary.performance_table = table( ...
        method, converged_runs, convergence_rate, avg_cpu, med_cpu, avg_outer, avg_major, avg_func, avg_logl, avg_policy_rmse, ...
        'VariableNames', {'Method','ConvergedRuns','ConvergenceRate','AvgCPU','MedianCPU', ...
                          'AvgOuterIter','AvgMajorIter','AvgFuncCount','AvgLogLik','AvgPolicyRMSE'});

    summary.avg_initializer_rmse = mean(results.initializer.ccp_rmse);
    summary.avg_initializer_time = mean(results.initializer.time);
    summary.avg_occupied_share = mean(results.initializer.occupied_share);
end

function [m, s, rmse, b] = summarize_theta(theta_mat, theta_true, converged)
    theta_use = theta_mat(:, converged);
    if isempty(theta_use)
        m = nan(size(theta_true));
        s = nan(size(theta_true));
        rmse = nan(size(theta_true));
        b = nan(size(theta_true));
        return
    end
    m = mean(theta_use, 2);
    s = std(theta_use, 0, 2);
    b = m - theta_true;
    rmse = sqrt(mean((theta_use - theta_true).^2, 2));
end

function x = mean_or_nan(v, mask)
    if nargin<2
        mask = true(size(v));
    end
    if any(mask)
        x = mean(v(mask));
    else
        x = NaN;
    end
end

function x = median_or_nan(v, mask)
    if nargin<2
        mask = true(size(v));
    end
    if any(mask)
        x = median(v(mask));
    else
        x = NaN;
    end
end

function plot_representative_surfaces(rep)
    figure('Color',[1 1 1]);

    subplot(2,2,1);
    zurcher2d.plot_policy_surface(rep.mp_true, rep.pk_true, 'True replacement probability');

    subplot(2,2,2);
    zurcher2d.plot_policy_surface(rep.mp_true, rep.pk_init, 'Initial 2D frequency CCP');

    subplot(2,2,3);
    zurcher2d.plot_policy_surface(rep.mp_true, rep.pk_npl, 'NPL: fixed-point CCP at estimated parameter');

    subplot(2,2,4);
    zurcher2d.plot_policy_surface(rep.mp_true, rep.pk_nfxp, 'NFXP: Bellman CCP at estimated parameter');
end
