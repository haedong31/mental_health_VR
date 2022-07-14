function [RR,ENT,Mean] = heterorecurrence2(IFS_address,cate_state,o1,o2)
    idx1 = find(cate_state == o1);
    if ~isempty(idx1)
        idx2 = find(cate_state(idx1) == o2);
    
        RR = power(length(idx2),2)/power(length(cate_state),2);
        r = cerecurr_y(IFS_address(idx2+1,:));
        rr = triu(r,1);
        dist = rr(:);
        dist(dist==0) = [];
        if isempty(dist)
            Mean = 0;
            ENT = 0;
        else
            count = histcounts(dist,'BinMethod','auto');
            Mean = mean(dist);
            prob = count/sum(count);
            nonz = prob(prob~=0);
            ENT = sum (nonz .* (-log2 (nonz)));
        end
    else
        RR = 0; ENT = 0; Mean = 0;
    end
end

function buffer = cerecurr_y(signal)
    N = size(signal,1);
    Y = signal;
    buffer=zeros(N);
    
    %h = waitbar(0,'Please wait...');
    for i=1:N
        %waitbar(i/N);
        x0=i;
        for j=i:N
            y0=j;
            % Calculate the euclidean distance
            distance = norm(Y(i,:)-Y(j,:));
            % Store the minimum distance between the two points
            buffer(x0,y0) = distance;
            buffer(y0,x0) = distance;        
        end
    end
    %close(h);
end
    