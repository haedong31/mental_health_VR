%% IFS 1st order plots
clc
clearvars
close all

data_dir1 = "./data/audio/control_embd3";
data_dir2 = "./data/audio/dementia_embd3";

save_dir1 = "./data/audio/ifs_control";
save_dir2 = "./data/audio/ifs_dementia";
mkdir(save_dir1)
mkdir(save_dir2)

dir_info1 = dir(data_dir1);
dir_info2 = dir(data_dir2);

fnames1 = {dir_info1.name};
fnames1 = fnames1(~ismember(fnames1,{'.','..'}));
fnames2 = {dir_info2.name};
fnames2 = fnames2(~ismember(fnames2,{'.','..'}));

n1 = length(fnames1);
n2 = length(fnames2);

num_id = 8;
alpha = 0.25;

for i=1:n1
    fpath = fullfile(data_dir1, fnames1(i));
    y = table2array(readtable(fpath));
    
    ot = OcTree(y,'binCapacity',n1/3,'maxDepth',2,'style','weighted');
    int_sig = OcTree_state(ot,y,n1);    
    ifs_coord = IFS(int_sig,num_id,alpha);
    
    figure('Color','w','Position',[100,100,430,400])
    plot(ifs_coord(:,1),ifs_coord(:,2),'.','Color','black')
    ylim([-1.5,1.5])
    xlim([-1.5,1.5])
    axis off
    
    [~,fname_stem,~] = fileparts(fnames1(i));
    saveas(gcf,fullfile(save_dir1,strcat(fname_stem,".png")))
    close
end

for i=1:n2
    fpath = fullfile(data_dir2, fnames2(i));
    y = table2array(readtable(fpath));
    
    ot = OcTree(y,'binCapacity',n2/3,'maxDepth',2,'style','weighted');
    int_sig = OcTree_state(ot,y,n2);    
    ifs_coord = IFS(int_sig,num_id,alpha);
    
    figure('Color','w','Position',[100,100,430,400])
    plot(ifs_coord(:,1),ifs_coord(:,2),'.','Color','black')
    ylim([-1.5,1.5])
    xlim([-1.5,1.5])
    axis off
    
    [~,fname_stem,~] = fileparts(fnames2(i));
    saveas(gcf,fullfile(save_dir2,strcat(fname_stem,".png")))
    close
end
%% Time-delay embedding
clc
clearvars
close all

data_dir = dir('./data/audio/dementia_wav/*.wav');
save_dir = "./data/audio/dementia_embd3";
num_files = length(data_dir);

downsample_factor = 1000;

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
    y_sampled = downsample(y,downsample_factor);

    mi = zeros(max_tau,1);
    for j=1:max_tau
        mi(j) = mutual_info(y_sampled,j,nbins);
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

    [embd_y,~] = time_delay_embed(y_sampled,tau(i),3);
    embd_y = array2table(embd_y,'VariableNames',{'dim1','dim2','dim3'});

    [~,fname_stem,~] = fileparts(data_dir(i).name);
    writetable(embd_y,fullfile(save_dir,strcat(fname_stem,'.csv')))
end