function [fnn_ratios] = FNN(x, tau, dim, r, tol)
    % x [vector]; input signal   
    % tau [integer]; time lag
    % dim [vector]; embedding dimension sizes
    % r [integer]; rth nearest neighbor
    % tol [numeric]; FNN-critetion tolerance

    xlen = length(x);

    % 1st neighborhood info
    idxs = zeros(1, xlen);
    old_dists = zeros(1, xlen);
    for i=1:xlen
        entire_nbhd = x;
        entire_nbhd(i) = [];
        [kidx, kdist] = knnsearch(entire_nbhd, x(i), 'K',r, 'Distance','euclidean');

        if (kidx < i)
            idxs(i) = kidx(r);
        else
            idxs(i) = kidx(r) + 1;
        end

        old_dists(i) = kdist(r)^2;
    end

    fnn_ratios = zeros(1, length(dim));
    for m=1:length(dim)
        fprintf('Dimension %i \n', m)
        fnn_cnts = 0;

        % calculate new distances
        xlen_next = xlen - m*tau; % remove last a few elements out of range
        new_dists = zeros(1, xlen_next);
        for i=1:xlen_next
            new_idx = i + m*tau;

            entire_nbhd = x;
            entire_nbhd(new_idx) = [];
            [~, kdist] = knnsearch(entire_nbhd, x(new_idx), 'K',r, 'Distance','euclidean');
            
            new_dists(i) = old_dists(i) + kdist(r)^2;

            % FNN criterion (FNN percentage)
            crit = power((new_dists(i) - old_dists(i))/old_dists(i), 1/2);
            if (crit > tol)
                fnn_cnts = fnn_cnts + 1;
            end
        end
        fnn_ratios(m) = fnn_cnts / xlen_next;
        old_dists = new_dists;
    end
end
