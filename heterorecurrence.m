function [RR,ENT,Mean,num_rgroup] = heterorecurrence(IFS_address,cate_state,Order)
    % input:
    % IFS_address - 2D coordinate of IFS
    % cate_state - corresponding state series of categorical variables
    % Order - particular categorical variable
    % nbins - number of bins to compute the ENT

    % output:
    % RR - heterogeneous recurrence rate
    % ENT - heterogeneous ENT
    % Mean - heterogeneous Mean

    %Author: Hui Yang
    %Affiliation: 
        %The Pennsylvania State University
        %310 Leohard Building, University Park, PA
        %Email: yanghui@gmail.com
        
    % If you find this toolbox useful, please cite the following paper:
    % [1]	H. Yang and Y. Chen, �Heterogeneous recurrence monitoring and control 
    % of nonlinear stochastic processes,� Chaos, Vol. 24, No. 1, p013138, 2014, 
    % DOI: 10.1063/1.4869306
    % [2]	Y. Chen and H. Yang, �Heterogeneous recurrence representation and quantification
    % of dynamic transitions in continuous nonlinear processes,� European Physical Journal B, 
    % Vol. 89, No. 6, p1-11, 2016, DOI: 10.1140/epjb/e2016-60850-y

    num_rgroup = 0;
    Index = find(cate_state==Order);
    if ~isempty(Index)
        num_rgroup = num_rgroup+1;
        RR = power(length(Index),2)/power(length(cate_state),2);
        r = cerecurr_y(IFS_address(Index+1,:));
        rr = triu(r,1);
        dist = rr(:);
        dist(dist==0) = [];

        if isempty(dist)
            Mean = 0;
            ENT = 0;
        else
            count = histcounts(dist,'BinMethod','auto');
            prob = count/sum(count);
            nonz = prob(prob~=0);
            ENT = sum (nonz .* (-log2 (nonz)));
            Mean = mean(dist);
        end
    else
        RR = 0; 
        ENT = 0; 
        Mean = 0;
    end
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
    