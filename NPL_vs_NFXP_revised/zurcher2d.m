classdef zurcher2d
% ZURCHER2D
% Two-dimensional extension of Rust's engine replacement model.
%
% State:
%   s = (x1, x2)
% where
%   x1 = mileage-like accumulated usage
%   x2 = slow wear/condition index
%
% Action:
%   d = 0 keep
%   d = 1 replace
%
% Dynamics:
%   Keep:    x1' = min(x1 + dx1, n1-1),  x2' = min(x2 + dx2, n2-1)
%   Replace: x1' = min(dx1,     n1-1),   x2' = min(dx2,     n2-1)
%
% Utility:
%   u_keep(s)    = -0.001 * (c1*x1 + c2*x2)
%   u_replace(s) = -RC
%
% The replacement transition equals the keep transition from the lowest
% state (0,0), so the expected-value representation keeps the same logic as
% in the 1D code: the replacement continuation value uses ev(1).

methods (Static)

    function mp = setup(mpopt)
        mp.n1 = 50;
        mp.n2 = 8;

        mp.p1 = [0.0937 0.4475 0.4459 0.0127]';
        mp.p2 = 0.85;

        mp.RC = 10.80;
        mp.c1 = 120.0;
        mp.c2 = 650.0;
        mp.beta = 0.995;

        mp.bellman_type = 'iv';
        mp.pnames_u = {'RC','c1','c2'};
        mp.pnames_P = {};

        mp.ap = dpsolver.setup();

        if nargin>0
            pfields = fieldnames(mpopt);
            for i=1:numel(pfields)
                mp.(pfields{i}) = mpopt.(pfields{i});
            end
        end

        mp.p1 = mp.p1(:);
        mp.p2 = mp.p2(:);

        mp.n = mp.n1 * mp.n2;
        mp.grid1 = (0:mp.n1-1)';
        mp.grid2 = (0:mp.n2-1)';

        [x1g, x2g] = ndgrid(mp.grid1, mp.grid2);
        mp.x1 = x1g(:);
        mp.x2 = x2g(:);
        mp.grid = [mp.x1 mp.x2];
    end

    function mp = set_theta(mp, theta)
        mp = vec2struct(theta, mp.pnames_u, mp);
    end

    function [u, du] = u(mp)
        u = zeros(mp.n, 2);
        u(:,1) = -0.001*(mp.c1*mp.x1 + mp.c2*mp.x2);
        u(:,2) = -mp.RC;

        if nargout>1
            n_u = numel(struct2vec(mp, mp.pnames_u));
            du = zeros(mp.n, n_u, 2);

            for iP = 1:numel(mp.pnames_u)
                pname = mp.pnames_u{iP};
                switch pname
                    case 'RC'
                        du(:,iP,1) = 0;
                        du(:,iP,2) = -1;
                    case 'c1'
                        du(:,iP,1) = -0.001*mp.x1;
                        du(:,iP,2) = 0;
                    case 'c2'
                        du(:,iP,1) = -0.001*mp.x2;
                        du(:,iP,2) = 0;
                    otherwise
                        error('Unknown utility parameter %s', pname);
                end
            end
        end
    end

    function P = statetransition(mp)
        [P1_keep, P1_rep] = zurcher2d.transition_1d(mp.p1, mp.n1);
        [P2_keep, P2_rep] = zurcher2d.transition_1d(mp.p2, mp.n2);

        % With column-major vectorization of an ndgrid state array, the full
        % row-stochastic transition matrix is kron(P_dim2, P_dim1).
        P = cell(2,1);
        P{1} = kron(P2_keep, P1_keep);  % keep
        P{2} = kron(P2_rep,  P1_rep);   % replace
    end

    function [Pkeep, Prep] = transition_1d(pvec, n)
        pfull = [pvec(:); max(0, 1-sum(pvec(:)))];
        m = numel(pfull);

        Pkeep = sparse(n,n);
        for i = 0:m-1
            rows = 1:(n-i);
            cols = (1+i):n;
            Pkeep = Pkeep + sparse(rows, cols, ones(1, n-i)*pfull(i+1), n, n);
            Pkeep(n-i, n) = 1 - sum(pfull(1:i));
        end
        Pkeep = sparse(Pkeep);

        Prep = sparse(n,n);
        for i = 1:m
            Prep(:,i) = pfull(i);
        end
    end

    function [V1, pk, dBellman_dV] = bellman(V0, mp, u, P)
        if numel(V0)==1
            V0 = zeros(mp.n,1);
        end
        if nargin<3 || isempty(u)
            u = zurcher2d.u(mp);
        end
        if nargin<4 || isempty(P)
            P = zurcher2d.statetransition(mp);
        end

        switch lower(mp.bellman_type)
            case 'iv'
                [V1, pk, dBellman_dV] = zurcher2d.bellman_iv(V0, mp, u, P);
            case 'ev'
                [V1, pk, dBellman_dV] = zurcher2d.bellman_ev(V0, mp, u, P);
            otherwise
                error('mp.bellman_type must be ''iv'' or ''ev''.');
        end
    end

    function [V1, pk, dBellman_dV] = bellman_iv(V0, mp, u, P)
        vK = u(:,1) + mp.beta*(P{1}*V0);
        vR = u(:,2) + mp.beta*(P{2}*V0);

        maxV = max(vK, vR);
        V1 = maxV + log(exp(vK-maxV) + exp(vR-maxV));

        if nargout>1
            pk = 1 ./ (1 + exp(vR-vK));
            pk = min(max(pk, 1e-12), 1-1e-12);
        end
        if nargout>2
            dBellman_dV = mp.beta * (spdiags(pk,0,mp.n,mp.n)*P{1} + spdiags(1-pk,0,mp.n,mp.n)*P{2});
        end
    end

    function [ev1, pk, dBellman_dev] = bellman_ev(ev, mp, u, P)
        if numel(ev)==1
            ev = zeros(mp.n,1);
        end

        vK = u(:,1) + mp.beta*ev;
        vR = u(:,2) + mp.beta*ev(1);

        maxV = max(vK, vR);
        V = maxV + log(exp(vK-maxV) + exp(vR-maxV));
        ev1 = P{1}*V;

        if nargout>1
            pk = 1 ./ (1 + exp(vR-vK));
            pk = min(max(pk, 1e-12), 1-1e-12);
        end
        if nargout>2
            dBellman_dev = mp.beta*(P{1}*spdiags(pk,0,mp.n,mp.n));
            dBellman_dev(:,1) = dBellman_dev(:,1) + mp.beta*P{1}*(1-pk);
        end
    end

    function [V, pk, dBellman_dV] = solve_model(mp, P, V0)
        if nargin<2 || isempty(P)
            P = zurcher2d.statetransition(mp);
        end
        if nargin<3
            V0 = 0;
        end
        u = zurcher2d.u(mp);
        bellman = @(V) zurcher2d.bellman(V, mp, u, P);
        [V, pk, dBellman_dV] = dpsolver.poly(bellman, V0, mp.ap, mp.beta);
    end

    function [f, g, h] = ll(data, mp, theta)
        global zurcher2d_V0

        y_j = [(1-data.d) data.d];
        mp = vec2struct(theta, mp.pnames_u, mp);

        if ~isempty(mp.pnames_P)
            error('zurcher2d.ll only supports stage-2 utility estimation with fixed transition probabilities.');
        end

        P = zurcher2d.statetransition(mp);

        if nargout>=2
            [u, du] = zurcher2d.u(mp);
        else
            u = zurcher2d.u(mp);
        end

        if isempty(zurcher2d_V0)
            zurcher2d_V0 = 0;
        end

        bellman = @(V) zurcher2d.bellman(V, mp, u, P);
        [zurcher2d_V0, pk, dBellman_dV] = dpsolver.poly(bellman, zurcher2d_V0, mp.ap, mp.beta);

        px_j = [pk(data.x) 1-pk(data.x)];
        px_j = min(max(px_j, 1e-12), 1-1e-12);

        logl = log(sum(y_j .* px_j, 2));
        f = mean(-logl);

        if nargout>=2
            dbellman = bsxfun(@times, du(:,:,1), pk) + bsxfun(@times, du(:,:,2), 1-pk);

            if strcmpi(mp.bellman_type, 'ev')
                dbellman = P{1}*dbellman;
            end

            dV = (speye(size(dBellman_dV)) - dBellman_dV) \ dbellman;

            score = zeros(size(logl,1), size(dbellman,2));
            for j = 1:2
                dv = du(:,:,j) + mp.beta*P{j}*dV;
                score = score + bsxfun(@times, (y_j(:,j)-px_j(:,j)), dv(data.x,:));
            end
            g = mean(-score, 1)';
        end

        if nargout>=3
            h = (score'*score) / size(logl,1);
        end
    end

    function data = simdata(N, T, mp, P, pk, seed, initdist)
        if nargin>=6 && ~isempty(seed)
            rng(seed, 'twister');
        end
        if nargin<7 || isempty(initdist)
            initdist = ones(mp.n,1) / mp.n;
        end

        p1full = [mp.p1(:); max(0, 1-sum(mp.p1(:)))];
        p2full = [mp.p2(:); max(0, 1-sum(mp.p2(:)))];

        cdf1 = cumsum(p1full);
        cdf2 = cumsum(p2full);
        cdf_init = cumsum(initdist(:));
        cdf_init(end) = 1;

        id = repmat((1:N), T, 1);
        t  = repmat((1:T)', 1, N);

        u_init = rand(1,N);
        x = zeros(T,N);
        for i=1:N
            x(1,i) = find(cdf_init >= u_init(i), 1, 'first');
        end

        u_dx1 = rand(T,N);
        u_dx2 = rand(T,N);
        u_d   = rand(T,N);

        dx1 = zeros(T,N);
        dx2 = zeros(T,N);
        for k=1:numel(mp.p1)
            dx1 = dx1 + (u_dx1 > cdf1(k));
        end
        for k=1:numel(mp.p2)
            dx2 = dx2 + (u_dx2 > cdf2(k));
        end

        s1 = zeros(T,N);
        s2 = zeros(T,N);
        [s1(1,:), s2(1,:)] = ind2sub([mp.n1, mp.n2], x(1,:));

        x_next = zeros(T,N);
        s1_next = zeros(T,N);
        s2_next = zeros(T,N);
        d = zeros(T,N);

        for it = 1:T
            pk_now = pk(x(it,:)');
            d(it,:) = (u_d(it,:) < (1 - pk_now')) * 1;

            s1_cur = s1(it,:) - 1;
            s2_cur = s2(it,:) - 1;

            s1_next(it,:) = min((1-d(it,:)).*(s1_cur + dx1(it,:)) + d(it,:).*dx1(it,:), mp.n1-1) + 1;
            s2_next(it,:) = min((1-d(it,:)).*(s2_cur + dx2(it,:)) + d(it,:).*dx2(it,:), mp.n2-1) + 1;

            x_next(it,:) = sub2ind([mp.n1, mp.n2], s1_next(it,:), s2_next(it,:));

            if it < T
                x(it+1,:) = x_next(it,:);
                s1(it+1,:) = s1_next(it,:);
                s2(it+1,:) = s2_next(it,:);
            end
        end

        data = struct();
        data.id = id;
        data.t = t;
        data.d = d;
        data.x = x;
        data.x_next = x_next;
        data.s1 = s1 - 1;
        data.s2 = s2 - 1;
        data.s1_next = s1_next - 1;
        data.s2_next = s2_next - 1;
        data.dx1 = dx1;
        data.dx2 = dx2;

        fields = fieldnames(data);
        for i=1:numel(fields)
            data.(fields{i}) = reshape(data.(fields{i}), T*N, 1);
        end
        data = struct2table(data);
    end

    function mp_hat = estimate_transitions(data, mp_template)
        mp_hat = mp_template;

        support1 = (numel(mp_template.p1)+1);
        counts1 = accumarray(double(data.dx1)+1, 1, [support1, 1], @sum, 0);
        mp_hat.p1 = counts1(1:end-1) / sum(counts1);

        support2 = (numel(mp_template.p2)+1);
        counts2 = accumarray(double(data.dx2)+1, 1, [support2, 1], @sum, 0);
        mp_hat.p2 = counts2(1:end-1) / sum(counts2);
    end

    function [pk0, info] = ccp_init_nonparam(data, mp, initopt)
        % Direct 2D extension of original case 4 in run_npl_sim.m:
        % estimate state-specific keep probabilities by raw empirical
        % frequencies, with no smoothing. Because the 2D state space is much
        % sparser, empty states are filled by the sample-wide keep frequency
        % so the initializer remains well-defined.
        if nargin<3
            initopt = struct();
        end
        if ~isfield(initopt, 'clip'); initopt.clip = 1e-6; end
        if ~isfield(initopt, 'empty_fill'); initopt.empty_fill = 'overall_keep'; end
        if ~isfield(initopt, 'empty_value'); initopt.empty_value = []; end

        idx = [double(data.s1)+1, double(data.s2)+1];
        count_grid = accumarray(idx, 1, [mp.n1, mp.n2], @sum, 0);

        keep_mask = (double(data.d)==0);
        if any(keep_mask)
            keep_grid = accumarray(idx(keep_mask,:), 1, [mp.n1, mp.n2], @sum, 0);
        else
            keep_grid = zeros(mp.n1, mp.n2);
        end

        pk_grid = nan(mp.n1, mp.n2);
        occupied = (count_grid > 0);
        pk_grid(occupied) = keep_grid(occupied) ./ count_grid(occupied);

        if any(~occupied(:))
            if ~isempty(initopt.empty_value)
                fill_value = initopt.empty_value;
            else
                switch lower(initopt.empty_fill)
                    case 'overall_keep'
                        fill_value = mean(double(data.d)==0);
                    case 'half'
                        fill_value = 0.5;
                    otherwise
                        error('Unknown empty-state fill rule: %s', initopt.empty_fill);
                end
            end
            if isnan(fill_value)
                fill_value = 0.5;
            end
            pk_grid(~occupied) = fill_value;
        else
            fill_value = NaN;
        end

        pk0 = pk_grid(:);
        pk0(pk0==0) = initopt.clip;
        pk0(pk0==1) = 1-initopt.clip;
        pk0 = min(max(pk0, initopt.clip), 1-initopt.clip);

        info = struct();
        info.method = 'frequency';
        info.occupied_share = mean(occupied(:));
        info.empty_share = mean(~occupied(:));
        info.count_grid = count_grid;
        info.keep_grid = keep_grid;
        info.pk_grid = pk_grid;
        info.empty_fill = fill_value;
    end

    function pp = eqb(mp, P, pk)
        pl = spdiags(pk,0,mp.n,mp.n)*P{1} + spdiags(1-pk,0,mp.n,mp.n)*P{2};

        A = [speye(mp.n) - pl', ones(mp.n,1); ones(1,mp.n), 0];
        b = [zeros(mp.n,1); 1];
        sol = A \ b;
        pp = sol(1:mp.n);
        pp = max(pp, 0);
        pp = pp / sum(pp);
    end

    function plot_policy_surface(mp, pk, ttl)
        Z = reshape(1-pk, mp.n1, mp.n2);
        surf(mp.grid2, mp.grid1, Z, 'EdgeColor','none');
        xlabel('State 2');
        ylabel('State 1');
        zlabel('Replacement probability');
        title(ttl);
        view(45,35);
        colorbar;
        axis tight;
        grid on;
    end

end
end
