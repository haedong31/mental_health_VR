clc
close all
clear variables

con_dir = dir('./data/audio/control/*.wav');
exp_dir = dir('./data/audio/dementia/*.wav');

len_con = length(con_dir);
len_exp = length(exp_dir);

%% control group
% input arguments for tau
con_tau = zeros(len_con,1);
tau = 1:100; % tau to be tested
len_tau = length(tau);
nbins = 2;

% input arguments for embedding dimension
con_dim = zeros(len_con,1);
dim = 1:10; % dimensions to be tested
r = 1;
tol = 5;

con_embd_mx = cell(len_con,1);
for i = 1:len_con
    % average left and right channels
    [y, fs] = audioread(strcat(con_dir(i).folder, '/', con_dir(i).name));
    y = mean(y,2);
    
    disp('Calculating tau')
    mi = zeros(len_tau, 1);
    for j = 1:len_tau
        mi(j) = mutual_info(y, tau(j), nbins);
    end

    % test
    if any(isnan(mi))
        fprintf('NaN MI at %i \n', i)
    end

    % select knee-point tau 
    knee_tau = knee_pt(mi);
    if mi(knee_tau+1) < mi(knee_tau)
        con_tau(i) = knee_tau+1;
    else
        con_tau(i) = knee_tau;
    end
    fprintf('Tau at %i: %f', i, con_tau(i))

    disp('Calculating FNN embedding dimension')
    fnn_ratios = FNN(y, con_tau(i), dim, r, tol);

    % select knee-point dim
    knee_dim = knee_pt(fnn_ratios);
    if fnn_ratios(knee_dim+1) < fnn_ratios(knee_dim)
        con_dim(i) = knee_dim + 1;
    else
        con_dim(i) = knee_dim;
    end
    fprintf('Dim at %i: %f', i, con_dim(i))

    % embedding
    disp('Calculating embedding matrix')
    [embd_y, ~] = time_delay_embed(y, con_tau(i), con_dim(i));
    con_embd_mx{i} = embd_y;
end

%% experimental group
% input arguments for tau
exp_tau = zeros(len_exp,1);
tau = 1:100; % tau to be tested
len_tau = length(tau);
nbins = 2;

% input arguments for embedding dimension
exp_dim = zeros(len_exp,1);
dim = 1:10; % dimensions to be tested
r = 1;
tol = 5;

exp_embd_mx = cell(len_exp,1);
for i = 1:len_exp
    % average left and right channels
    [y, fs] = audioread(strcat(exp_dir(i).folder, '/', exp_dir(i).name));
    y = mean(y,2);
    
    disp('Calculating tau')
    mi = zeros(len_tau, 1);
    for j = 1:len_tau
        mi(j) = mutual_info(y, tau(j), nbins);
    end

    % test
    if any(isnan(mi))
        fprintf('NaN MI at %i \n', i)
    end

    % select knee-point tau 
    knee_tau = knee_pt(mi);
    if mi(knee_tau+1) < mi(knee_tau)
        exp_tau(i) = knee_tau+1;
    else
        exp_tau(i) = knee_tau;
    end
    fprintf('Tau at %i: %f', i, exp_tau(i))

    disp('Calculating FNN embedding dimension')
    fnn_ratios = FNN(y, exp_tau(i), dim, r, tol);

    % select knee-point dim
    knee_dim = knee_pt(fnn_ratios);
    if fnn_ratios(knee_dim+1) < fnn_ratios(knee_dim)
        exp_dim(i) = knee_dim + 1;
    else
        exp_dim(i) = knee_dim;
    end
    fprintf('Dim at %i: %f', i, exp_dim(i))

    % embedding
    disp('Calculating embedding matrix')
    [embd_y, ~] = time_delay_embed(y, exp_tau(i), exp_dim(i));
    exp_embd_mx{i} = embd_y;
end

%% save
save('time_delay_embd.mat', 'con_tau', 'exp_tau', 'con_dim', 'exp_dim', 'con_embd_mx','exp_embd_mx')
