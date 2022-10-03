clc
clearvars
close all

data_dir = dir('./data/audio/dementia_wav/*.wav');
save_dir = "./data/audio/dementia_embd3";
num_files = length(data_dir);

% hyperparameters for searching tau
tau = zeros(num_files,1);
max_tau = 20;
nbins = 2;

for i=1:num_files
    % read .wav file
    fpath = fullfile(data_dir(i).folder,data_dir(i).name);
    [y,fs] = audioread(fpath);
    
    % average out left and right channels
    y = mean(y,2);

    mi = zeros(max_tau,1);
    for j=1:max_tau
        mi(j) = mutual_info(y,j,nbins);
    end

    if any(isnan(mi))
        fprintf('NaN MI at %i \n', i)
    end

    knee_tau = knee_pt(mi);
    if mi(knee_tau+1) < mi(knee_tau)
        tau(i) = knee_tau+1;
    else
        tau(i) = knee_tau;
    end

    [embd_y,~] = time_delay_embed(y,tau(i),3);
    embd_y = array2table(embd_y,'VariableNames',{'dim1','dim2','dim3'});

    [~,fname_stem,~] = fileparts(data_dir(i).name);
    writetable(embd_y,fullfile(save_dir,strcat(fname_stem,'.csv')))
end