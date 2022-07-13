function [IFS_address] = IFS(cate_state,KK,alpha)

%input:
%cate_state - state series of categorical variables     
%alpha - IFS parameter 	1

%output:
%IFS_address - 2D coordinate of IFS

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



x_address(1) = 0; y_address(1)=  0;
for i=1:length(cate_state)
    x_address(i+1) = alpha*x_address(i)+cos(cate_state(i)*2*pi/KK);
    y_address(i+1) = alpha*y_address(i)+sin(cate_state(i)*2*pi/KK);
end

IFS_address = [x_address',y_address'];
