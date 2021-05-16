function mi =  mutual_info(x, tau, nbins)
    % tau [vector]; step size of time index
    % nbin [integer]; number of bins

    [len, ~] = size(x);

    % extract x_tau and align sizes of x and x_tau
    x_tau = x(1+tau:len);
    x(len-(tau-1):end) = [];
    
    % probability distributions
    joint_freq_mx = hist3([x, x_tau], 'Nbins',[nbins, nbins]);
    
    pjoint = joint_freq_mx ./ sum(joint_freq_mx, 'all');
    
    xfreq = histcounts(x, nbins);
    px = xfreq ./ sum(xfreq);
    
    % entropy
    hx = -sum(px .* log(px));
    
    % conditional entropy
    [~, ncols] = size(joint_freq_mx);
    h_xly_elt = zeros(1, ncols);
    for i=1:ncols
        % p(x|y=x_tau(i))
        xly_freq = joint_freq_mx(:,i);
        p_xly = xly_freq ./ sum(xly_freq);
        
        % to avoid log(0); force them to be 0 (=log(1))
        p_xly(p_xly == 0) = 1;
        
        % p(x, y=x_tau(i)) 
        pjoint_col = pjoint(:,i);
        
        h_xly_elt(i) = sum(pjoint_col .* log(p_xly));
    end
    h_xly = -sum(h_xly_elt);
    
    mi = hx - h_xly;        
end
