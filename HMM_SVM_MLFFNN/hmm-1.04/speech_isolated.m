class1_file_loc = '../speech_dataset/isolated/23/3/';
class2_file_loc = '../speech_dataset/isolated/23/o/';
class3_file_loc = '../speech_dataset/isolated/23/z/';

D = dir(fullfile(class1_file_loc, '*.mfcc'));
A = cell(size(D));
class1_data = [];
seq_lens_1 = [];
for i=1:length(D)
      [class_data, seq_len] = get_class_data(strcat(class1_file_loc,D(i).name)); 
      class1_data = [class1_data; class_data];
      seq_lens_1 = [seq_lens_1 seq_len];
end
n_class1 = size(class1_data,1);

class_data = [];
D = dir(fullfile(class2_file_loc, '*.mfcc'));
A = cell(size(D));
class2_data = [];
seq_lens_2 = [];
for i=1:length(D)
      [class_data, seq_len] = get_class_data(strcat(class2_file_loc,D(i).name)); 
      class2_data = [class2_data; class_data];
      seq_lens_2 = [seq_lens_2 seq_len];
end
n_class2 = size(class2_data,1);

class_data = [];
D = dir(fullfile(class3_file_loc, '*.mfcc'));
A = cell(size(D));
class3_data = [];
seq_lens_3 = [];
for i=1:length(D)
      [class_data, seq_len] = get_class_data(strcat(class3_file_loc,D(i).name)); 
      class3_data = [class3_data; class_data];
      seq_lens_3 = [seq_lens_3 seq_len];
end
n_class3 = size(class3_data,1);

data_all_classes = [class1_data; class2_data; class3_data];
cluster_index = quantize(data_all_classes);
cluster_index(:,1) = cluster_index(:,1)-1;

start_index1 = 1;
start_index2 = start_index1 + n_class1;
start_index3 = start_index2 + n_class2;
get_sequenced_data(cluster_index, start_index1, n_class1, seq_lens_1, 1);
get_sequenced_data(cluster_index, start_index2, n_class2, seq_lens_2, 2);
get_sequenced_data(cluster_index, start_index3, n_class3, seq_lens_3, 3);


function [class_data, seq_len] = get_class_data(filename)
    class_data = [];
    fileID = fopen(filename); 
    C = [];
    P = [];
    C = textscan(fileID,'%d %d \n',1);
    seq_len = C{1}*C{2};
    P = textscan(fileID, '%f', seq_len);
    class_data =  P{1};  
    fclose(fileID);
end

function idx = quantize(class_data)
    [idx, means] = kmeans(class_data, 5);
end

function get_sequenced_data(cluster_index, start_index, n_class, class_seq_lens, class_no)
    class_sequence = [];
    sample_ctr = 0;
    
    s_train = strcat('speech_train_class_seq', num2str(class_no));
    s_train = strcat(s_train, '.txt');
    
    s_val = strcat('speech_val_class_seq', num2str(class_no));
    s_val = strcat(s_val, '.txt');
    
    s_test = strcat('speech_test_class_seq', num2str(class_no));
    s_test = strcat(s_test, '.txt');
    
    fileID_train = fopen(s_train,'w');
    fileID_val = fopen(s_val,'w');
    fileID_test = fopen(s_test,'w');
    
    for i = start_index : start_index+n_class-1
        class_sequence = [class_sequence cluster_index(i)];
    end
    start_index_single_seq = 1;
    for j = 1:size(class_seq_lens, 2)
        sample_ctr = sample_ctr + 1;
        sequence = [];
        seq_len = class_seq_lens(j);
        sequence = class_sequence(start_index_single_seq:start_index_single_seq+seq_len-1);
        %write this sequence in file
         if (sample_ctr <= 39)
            fprintf(fileID_train, '%d ', sequence);
            fprintf(fileID_train, '\n');
        end
        
        if (sample_ctr > 39 && sample_ctr < 50)
            fprintf(fileID_val, '%d ', sequence);
            fprintf(fileID_val, '\n');   
        end
        
        if (sample_ctr >= 50)
            fprintf(fileID_test, '%d ', sequence);
            fprintf(fileID_test, '\n');   
        end
        start_index_single_seq = start_index_single_seq+seq_len;
    end
    fclose(fileID_train);
    fclose(fileID_val);
    fclose(fileID_test);
    
end
