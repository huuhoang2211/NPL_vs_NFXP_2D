classdef npl2d
% NPL2D
% 2D nested pseudo likelihood estimator built to mirror the original 1D NPL
% architecture more closely:
%   * for each outer iteration K, precompute Finv = inv(I-beta*Fu(pk_{K-1}))
%     once and pass it into the inner pseudo-likelihood;
%   * avoid redundant utility evaluations by computing u once per inner
%     likelihood call and passing it into phi() and lambda().

methods (Static)

    function pk = Psi(mp, pk0, P, u)
        if nargin<4 || isempty(u)
            u = zurcher2d.u(mp);
        end
        Vsigma = npl2d.phi(mp, pk0, P, [], u);
        pk = npl2d.lambda(Vsigma, mp, P, u);
    end

    function Fu = controlled_transition(mp, pk, P)
        pk = min(max(pk(:), 1e-12), 1-1e-12);
        Fu = spdiags(pk,0,mp.n,mp.n)*P{1} + spdiags(1-pk,0,mp.n,mp.n)*P{2};
    end

    function Vsigma = phi(mp, pk, P, Finv, u)
        pk = min(max(pk(:), 1e-12), 1-1e-12);

        if nargin<4 || isempty(Finv)
            Fu = npl2d.controlled_transition(mp, pk, P);
            Finv = inv(speye(mp.n) - mp.beta*Fu);
        end
        if nargin<5 || isempty(u)
            u = zurcher2d.u(mp);
        end

        eulerc = 0.5772156649015328606065120900824024310421;

        vK = u(:,1) + eulerc - log(pk);
        vR = u(:,2) + eulerc - log(1-pk);
        pv = pk .* vK + (1-pk) .* vR;

        Vsigma = Finv * pv;
    end

    function pK = lambda(Vsigma, mp, P, u)
        if nargin<4 || isempty(u)
            u = zurcher2d.u(mp);
        end

        VK = u(:,1) + mp.beta*(P{1}*Vsigma);
        VR = u(:,2) + mp.beta*(P{2}*Vsigma);

        pK = 1 ./ (1 + exp(VR-VK));
        pK = min(max(pK, 1e-12), 1-1e-12);
    end

    function [f, g, h] = ll(theta, pk0, data, P, mp, Finv)
        global npl2d_pk_last

        mp = zurcher2d.set_theta(mp, theta);
        pk0 = min(max(pk0(:), 1e-12), 1-1e-12);

        if nargin<6 || isempty(Finv)
            Fu = npl2d.controlled_transition(mp, pk0, P);
            Finv = inv(speye(mp.n) - mp.beta*Fu);
        end

        [u, du] = zurcher2d.u(mp);

        Vsigma = npl2d.phi(mp, pk0, P, Finv, u);
        pk = npl2d.lambda(Vsigma, mp, P, u);
        npl2d_pk_last = pk;

        pKdata = pk(data.x);
        pKdata = min(max(pKdata, 1e-12), 1-1e-12);
        dk = double(data.d==0);
        dr = double(data.d==1);

        logl = log(pKdata.*dk + (1-pKdata).*dr);
        f = -mean(logl);

        if nargout>=2
            dP = P{1} - P{2};
            dpv = bsxfun(@times, du(:,:,1), pk0) + bsxfun(@times, du(:,:,2), 1-pk0);
            dVsigma = Finv * dpv;
            dDelta = du(:,:,1) - du(:,:,2) + mp.beta*(dP*dVsigma);

            res = dk - pKdata;
            score = bsxfun(@times, res, dDelta(data.x,:));

            g = -mean(score, 1)';
        end

        if nargout>=3
            h = (score'*score) / size(score,1);
        end
    end

    function [results, details] = estim(theta0, pk0, data, P, mp, Kmax, tol_theta, tol_pk, verbose)
        global npl2d_pk_last

        if nargin<6 || isempty(Kmax); Kmax = 100; end
        if nargin<7 || isempty(tol_theta); tol_theta = 1e-6; end
        if nargin<8 || isempty(tol_pk); tol_pk = 1e-8; end
        if nargin<9 || isempty(verbose); verbose = false; end

        options = optimset('Algorithm','trust-region', 'Display','off', ...
                           'GradObj','on', 'Hessian','on', ...
                           'TolFun',1e-8, 'TolX',1e-8, ...
                           'MaxIter',400, 'MaxFunEvals',5000);

        pk_old = min(max(pk0(:), 1e-8), 1-1e-8);
        theta_old = theta0(:);

        totalMajorIter = 0;
        totalFuncCount = 0;
        converged_outer = false;

        if verbose
            fprintf('------------------------------------------------------------\n');
            fprintf('NPL2D started\n');
            fprintf('------------------------------------------------------------\n');
        end

        I = speye(mp.n);
        t_outer = tic;
        for K = 1:Kmax
            Fu = npl2d.controlled_transition(mp, pk_old, P);
            Finv = inv(I - mp.beta*Fu);

            npl2d_pk_last = pk_old;
            [theta_new, FVAL, EXITFLAG, OUTPUT] = fminunc(@(theta) npl2d.ll(theta, pk_old, data, P, mp, Finv), ...
                                                          theta_old, options);

            totalMajorIter = totalMajorIter + OUTPUT.iterations;
            totalFuncCount = totalFuncCount + OUTPUT.funcCount;
            pk_new = npl2d_pk_last;

            theta_metric = max(abs(theta_new - theta_old));
            pk_metric = max(abs(pk_new - pk_old));

            if verbose
                fprintf('K=%2d | ll=%12.6f | theta=[%9.4f %9.4f %9.4f] | dtheta=%8.2e | dpk=%8.2e\n', ...
                    K, -FVAL*height(data), theta_new(1), theta_new(2), theta_new(3), theta_metric, pk_metric);
            end

            theta_old = theta_new;
            pk_old = pk_new;

            if theta_metric < tol_theta && pk_metric < tol_pk && EXITFLAG >= 1 && EXITFLAG <= 3
                converged_outer = true;
                break
            end
        end
        runtime = toc(t_outer);

        theta_hat = theta_old;
        mp_hat = zurcher2d.set_theta(mp, theta_hat);

        % Final pseudo-likelihood evaluation for Hessian, using the same
        % once-per-outer-iteration precomputed inverse logic.
        Fu = npl2d.controlled_transition(mp, pk_old, P);
        Finv = inv(I - mp.beta*Fu);
        [~, ~, h] = npl2d.ll(theta_hat, pk_old, data, P, mp, Finv);
        Avar = inv(h*height(data));

        pk_fix = npl2d.solve(mp_hat, P, pk_old);
        policy_rmse = NaN;
        if isfield(mp, 'pk_true')
            policy_rmse = sqrt(mean((pk_fix - mp.pk_true).^2));
        end

        results = npl2d.make_result_struct(theta_hat, mp.pnames_u);
        results.cputime = runtime;
        results.outerIter = K;
        results.MajorIter = totalMajorIter;
        results.funcCount = totalFuncCount;
        results.llval = -FVAL*height(data);
        results.converged = converged_outer;
        results.policy_rmse = policy_rmse;

        details = struct();
        details.theta_hat = theta_hat;
        details.mp_hat = mp_hat;
        details.pk_last = pk_old;
        details.pk_fix = pk_fix;
        details.Avar = Avar;
    end

    function pk_solve = solve(mp, P, pk0)
        pk_old = min(max(pk0(:), 1e-12), 1-1e-12);
        u = zurcher2d.u(mp);

        for i = 1:200
            pk_new = npl2d.Psi(mp, pk_old, P, u);
            if max(abs(pk_new - pk_old)) < 1e-12
                pk_old = pk_new;
                break
            end
            pk_old = pk_new;
        end
        pk_solve = pk_old;
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
