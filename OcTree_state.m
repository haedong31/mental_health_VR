function cate_state = OcTree_state(OT,X_Lorenz,N)

% input:
% OT - Octree structure
% X_Lorenz - Lorenz data
% N - size of Lorenz model

% output:
% cate_state - state series of categorical variables

%Author: Hui Yang
%Affiliation: 
       %The Pennsylvania State University
       %310 Leohard Building, University Park, PA
       %Email: yanghui@gmail.com
       
% If you find this toolbox useful, please cite the following paper:
% [1]	H. Yang and Y. Chen, “Heterogeneous recurrence monitoring and control 
% of nonlinear stochastic processes,” Chaos, Vol. 24, No. 1, p013138, 2014, 
% DOI: 10.1063/1.4869306
% [2]	Y. Chen and H. Yang, “Heterogeneous recurrence representation and quantification
% of dynamic transitions in continuous nonlinear processes,” European Physical Journal B, 
% Vol. 89, No. 6, p1-11, 2016, DOI: 10.1140/epjb/e2016-60850-y


K = unique(OT.PointBins);
KK = length(K);
cate_state = zeros(N,1);

for i=1:KK
    t = find(OT.PointBins==K(i));
    cate_state(t)=i;
    clear t
end

if nargout == 0
    figure('color','w')
    boxH = OT.plot;
    cols = lines(OT.BinCount);
    doplot3 = @(p,varargin)plot3(p(:,1),p(:,2),p(:,3),varargin{:});
    for i = 1:OT.BinCount
        set(boxH(i),'Color',cols(i,:),'LineWidth', 1+OT.BinDepths(i))
        doplot3(X_Lorenz(OT.PointBins==i,:),'.','Color',cols(i,:))
    end
    axis image, view(3)
end