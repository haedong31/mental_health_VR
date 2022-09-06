function feat_vec = heterorecurrence2(IFS_address,cate_state,num_id)
    rr = NaN(num_id,num_id);
    rent = NaN(num_id,num_id);
    rmean = NaN(num_id,num_id);
    num_rgroup = zeros(1,num_id);
    for i=1:num_id
        idx1 = find(cate_state == i);
        if isempty(idx1)
            rr(i,:) = 0;
            rent(i,:) = 0;
            rmean(i,:) = 0;
            continue
        end

        for j=1:num_id
            idx2 = find(cate_state(idx1) == j);
            num_rgroup(i) = num_rgroup(i)+1;
            rr(i,j) = power(length(idx2),2)/power(length(cate_state),2);
            
            rmx = cerecurr_y(IFS_address(idx2+1,:));
            trir = triu(rmx,1);
            flatr = trir(:);
            flatr(flatr==0) = [];
            if isempty(flatr)
                rent(i,j) = 0;
                rmean(i,j) = 0;
                continue
            end
            count = histcounts(flatr,'BinMethod','auto');
            prob = count/sum(count);
            nonz = prob(prob~=0);
            rent(i,j) = sum (nonz .* (-log2 (nonz)));
            rmean(i,j) = mean(flatr);
        end
    end
    feat_vec = [reshape(rr,1,[]), reshape(rent,1,[]), reshape(rmean,1,[]), num_rgroup];
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
    