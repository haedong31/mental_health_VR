clc
close all
clearvars

data_dir1 = fullfile(pwd,"data/hra/control/");
data_dir2 = fullfile(pwd,"data/hra/dementia/");

fnames_hc = dir(data_dir1);
fnames_hc = fnames_hc(~ismember({fnames_hc.name},{'.','..'}));
fnames_ad = dir(data_dir2);
fnames_ad = fnames_ad(~ismember({fnames_ad.name},{'.','..'}));

num_id = 32;

fsize1 = size(fnames_hc,1);
fsize2 = size(fnames_ad,1);
feat_mx = zeros(fsize1+fsize2,32*3+32*32*3);

alpha = 0.04;
nbins = 0;
order = 1;

for i=1:fsize1
    out = sprintf("HC File [%i/%i]", i, fsize2);
    disp(out);
    
    csig = readtable(fullfile(data_dir1,fnames_hc(i).name));
    csig = csig.x + 1;
    ifs_add = IFS(csig,num_id,alpha);
    
    % 1st order
    for j=1:num_id
        [feat_mx(i,j),feat_mx(i,j+num_id),feat_mx(i,j+num_id*2)] = ...
            heterorecurrence(ifs_add,csig,j,nbins);
        % 2nd order
        for k=1:num_id
            [feat_mx(i,k+num_id*(3*j)),feat_mx(i,k+num_id*(3*(j+1))),feat_mx(i,k+num_id*(3*(j+2)))] = ...
                heterorecurrence2(ifs_add,csig,j,k);
        end
    end
end

for i=1:fsize2
    out = sprintf("AD File [%i/%i]", i, fsize2);
    disp(out)

    csig = readtable(fullfile(data_dir2,fnames_ad(i).name));
    csig = csig.x + 1;
    ifs_add = IFS(csig,num_id,alpha);

    % 1st order
    for j=1:num_id
        [feat_mx(fsize1+i,j),feat_mx(fsize1+i,j+num_id),feat_mx(fsize1+i,num_id*2)] = ...
            heterorecurrence(ifs_add,csig,j,nbins);
        % 2nd order
        for k=1:num_id
            [feat_mx(fsize1+i,k+num_id*(3*j)),feat_mx(fsize1+i,k+num_id*(3*(j+1))),feat_mx(fsize1+i,k+num_id*(3*(j+2)))] = ...
                heterorecurrence2(ifs_add,csig,j,k);
        end
    end
end

writematrix(feat_mx,fullfile(pwd,"data/hra/hrq_mx.csv"));
