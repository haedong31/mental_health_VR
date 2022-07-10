clc
clearvars
close all

data_dir1 = fullfile(pwd,"data/hra/control/");
data_dir2 = fullfile(pwd,"data/hra/dementia/");

fnames_hc = dir(data_dir1);
fnames_hc = fnames_hc(~ismember({fnames_hc.name},{'.','..'}));
fnames_ad = dir(data_dir2);
fnames_ad = fnames_ad(~ismember({fnames_ad.name},{'.','..'}));

x1 = readtable(fullfile(data_dir1,fnames_hc(1).name));
x2 = readtable(fullfile(data_dir2,fnames_ad(1).name));
num_id = 32;

alpha = 0.02;
ifs_hc = IFS(x1.x+1,32,alpha);
ifs_ad = IFS(x2.x+1,32,alpha);
ifs_hc(1,:) = [];
ifs_ad(1,:) = [];

f = figure('Color','w');
hold on
for i=1:num_id
    id_idx = find(x1.x == (i-1));
    if i == num_id
        plot(ifs_hc(id_idx,1),ifs_hc(id_idx,2),'.','Color','g')
    else
        plot(ifs_hc(id_idx,1),ifs_hc(id_idx,2),'.','Color','b')
    end
    c = mean(ifs_hc(id_idx,:),1);
    text(c(1)+0.06,c(2)+0.06,num2str(i),'FontWeight','bold',...
        'VerticalAlignment','top','HorizontalAlignment','right')
end
hold off
title('IFS Plot of Control Participant1')
set(gca,'LineWidth',1.5,'FontWeight','bold')
