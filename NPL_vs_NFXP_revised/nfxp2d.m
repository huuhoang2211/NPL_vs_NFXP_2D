classdef nfxp2d
% NFXP2D
% 2D nested fixed-point estimator. Stage-2 only: transition probabilities
% are held fixed at the common first-stage estimates so that the comparison
% with NPL isolates the estimator, not the first-step transition fit.

methods (Static)

    function [results, details] = estim(data, mp, theta0, verbose)
        global zurcher2d_V0

        if nargin<3 || isempty(theta0)
            theta0 = zeros(numel(struct2vec(mp, mp.pnames_u)), 1);
        end
        if nargin<4 || isempty(verbose)
            verbose = false;
        end

        if ~isempty(mp.pnames_P)
            error('nfxp2d.estim currently supports only stage-2 utility estimation with fixed transitions.');
        end

        zurcher2d_V0 = 0;

        optim_options = optimset('Algorithm','trust-region', 'Display','off', ...
                                 'GradObj','on', 'TolFun',1e-8, 'TolX',1e-8, ...
                                 'Hessian','on', 'MaxIter',400, 'MaxFunEvals',5000);

        if verbose
            fprintf('------------------------------------------------------------\n');
            fprintf('NFXP2D started\n');
            fprintf('------------------------------------------------------------\n');
        end

        t_outer = tic;
        llfun = @(theta) zurcher2d.ll(data, mp, theta);
        [theta_hat, FVAL, EXITFLAG, OUTPUT] = fminunc(llfun, theta0, optim_options);
        runtime = toc(t_outer);

        [~, g, h] = llfun(theta_hat);
        Avar = inv(h*height(data));

        mp_hat = zurcher2d.set_theta(mp, theta_hat);
        [~, pk_hat] = zurcher2d.solve_model(mp_hat);

        policy_rmse = NaN;
        if isfield(mp, 'pk_true')
            policy_rmse = sqrt(mean((pk_hat - mp.pk_true).^2));
        end

        results = nfxp2d.make_result_struct(theta_hat, mp.pnames_u);
        results.grad_direc = g' * (h \ g);
        results.cputime = runtime;
        results.converged = (EXITFLAG >= 1 && EXITFLAG <= 3);
        results.MajorIter = OUTPUT.iterations;
        results.funcCount = OUTPUT.funcCount;
        results.llval = -FVAL * height(data);
        results.policy_rmse = policy_rmse;

        details = struct();
        details.theta_hat = theta_hat;
        details.Avar = Avar;
        details.mp_hat = mp_hat;
        details.pk_hat = pk_hat;

        if verbose
            fprintf('theta_hat = [%g, %g, %g]\n', theta_hat(1), theta_hat(2), theta_hat(3));
            fprintf('ll        = %g\n', results.llval);
            fprintf('time      = %g seconds\n', results.cputime);
        end
    end

    function res = make_result_struct(theta, pnames)
        res = struct();
        cursor = 1;
        for i=1:numel(pnames)
            res.(pnames{i}) = theta(cursor);
            cursor = cursor + 1;
        end
    end

end
end
