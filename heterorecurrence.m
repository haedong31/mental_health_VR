function feat_vec = heterorecurrence(ifs_address,cate_state,num_id)
    rr = NaN(1,num_id);
    rent = NaN(1,num_id);
    rmean = NaN(1,num_id);

    for i=1:num_id
        Index = find(cate_state==i);
        if isempty(Index)
            rr(i) = 0; 
            rent(i) = 0; 
            rmean(i) = 0;
            continue
        end

        rr(i) = power(length(Index),2)/power(length(cate_state),2);        
        rmx = cerecurr_y(ifs_address(Index+1,:));
        trir = triu(rmx,1);
        flatr = trir(:);
        flatr(flatr==0) = [];
        if isempty(flatr)
            rent(i) = 0;
            rmean(i) = 0;
            continue
        end
        count = histcounts(flatr,'BinMethod','auto');
        prob = count/sum(count);
        nonz = prob(prob~=0);
        rent(i) = sum (nonz .* (-log2 (nonz)));
        rmean(i) = mean(flatr);
    end
    feat_vec = [rr,rent,rmean];
end

function buffer = cerecurr_y(signal)
    %This program produces a recurrence plot of the, possibly multivariate,
    %data set. That means, for each point in the data set it looks for all 
    %points, such that the distance between these two points is smaller 
    %than a given size in a given embedding space. 
    
    %Author: Hui Yang
    %Affiliation: 
           %Department of Industrial Engineering and Management 
           %Oklahoma State University,Stillwater, 74075
           %Email: yanghui@gmail.com
    
    %input:
    %signal: input time series     
    %dim - embedded dimension 	1
    %tau - delay of the vectors 	1
    
    %output:
    %buffer - Matrix containing the pair distances.
    
    %example:
    %t=sin(-pi:pi/100:10*pi);
    %cerecurr_y(t,2,1);
    
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
    