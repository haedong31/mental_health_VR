clc
close all
clear variables

data_dir = dir('./data/audio/dementia/*.wav');
num_files = length(data_dir);

% input arguments for time dealy
t = zeros(num_files,1);
downsample_factor1 = 1000;
downsample_factor2 = 1000;
max_tau = 50;
nbins = 2;

% input arguments for embedding dimension
d = zeros(num_files,1);
max_dim = 20; % dimensions to be tested
r = 1;
tol = 5;

for i = (floor(num_files/2) + 1):num_files
    % average out left and right channels
    folder_name = data_dir(i).folder;
    audio_file_name = data_dir(i).name;
    [y, fs] = audioread(strcat(folder_name, '/', audio_file_name));
    y = mean(y,2);
    
    disp('Calculating tau')
    mi = zeros(max_tau, 1);
    for j = 1:max_tau
        mi(j) = mutual_info(y, j, nbins);
    end

    % test
    if any(isnan(mi))
        fprintf('NaN MI at %i \n', i)
    end

    % select knee-point tau 
    knee_tau = knee_pt(mi);
    if mi(knee_tau+1) < mi(knee_tau)
        t(i) = knee_tau+1;
    else
        t(i) = knee_tau;
    end

    disp('Calculating FNN embedding dimension')
    % downsample to reduce computational burden
    y_sampled = downsample(y, downsample_factor1);
    fnn_ratios = fnn(y_sampled, t(i), 1:max_dim, r, tol);

    % select knee-point dim
    knee_dim = knee_pt(fnn_ratios);
    if fnn_ratios(knee_dim+1) < fnn_ratios(knee_dim)
        d(i) = knee_dim + 1;
    else
        d(i) = knee_dim;
    end

    fprintf('Time delay: %i | Embedding dimension: %i \n', t(i), d(i));
    
    disp('Generating reccurence matrix')
    y_sampled = downsample(y, downsample_factor2);
    [embd_y, ~] = time_delay_embed(y_sampled, t(i), d(i));
    [num_pts, ~] = size(embd_y);
    
    rec_mx = pdist(embd_y,'euclidean');
    rec_mx = squareform(rec_mx);
    
    % recurrence plot
    rp_file_name = strsplit(audio_file_name, '.');
    rp_file_name = strcat(rp_file_name{1}, '.png');
    out_path = strcat(folder_name, '/', rp_file_name);
    
    figure('Position',[100, 100, 700, 700])
    imagesc(rec_mx)
    colormap jet
    axis image
    get(gcf, 'CurrentAxes')
    set(gca, 'YDir','normal')
    set(gca, 'Visible', 'off')
    
    saveas(gcf, out_path)
    close
end
