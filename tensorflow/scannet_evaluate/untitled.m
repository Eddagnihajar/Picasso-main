% write the ground truth of scannet


dataDir = '/media/huan/huanBook/reprocessed_data/ScanNet'; %/media/huan/MyBook/Datasets/ScanNet

% folder = {'scans_trainval','scans_test'};

writeDir = '/media/huan/huanBook/PycharmProjects/EnyaTrain/Visualization/ScanNet/GT';

files = textread(fullfile(dataDir,'scannetv2_val.txt'),'%s');
for k=1:length(files)
   labelPath = fullfile(dataDir,'scans_trainval',files{k},[files{k} '_semantic_labels.txt']); 
   gt_labels = load(labelPath); 
    
   writematrix(gt_labels, fullfile(writeDir,'val',[files{k} '.txt'])); 
end