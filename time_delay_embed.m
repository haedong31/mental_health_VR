function [embedX, xidx] = time_delay_embed(x, tau, n)
    % x [vector]; input signal

    xlen = length(x);
    
    % size of embedded vector
    k = xlen - (n-1)*tau;
    xidx = 1:k;

    embedX = zeros(k, n);
    for i=1:k
        row_idx = i + (0:(n-1)).*tau;
        embedX(i, :) = x(row_idx);
    end
end
