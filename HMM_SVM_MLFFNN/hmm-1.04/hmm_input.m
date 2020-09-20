[class1_data, class1_seq_lens] = get_class_data('../ocr_dataset/HandWritten_data/DATA/FeaturesHW/bA.ldf');
n_class1 = size(class1_data,1);

[class2_data, class2_seq_lens] = get_class_data('../ocr_dataset/HandWritten_data/DATA/FeaturesHW/dA.ldf');
n_class2 = size(class2_data,1);

[class3_data, class3_seq_lens] = get_class_data('../ocr_dataset/HandWritten_data/DATA/FeaturesHW/tA.ldf');
n_class3 = size(class3_data,1);

data_all_classes = [class1_data; class2_data; class3_data];
cluster_index = quantize(data_all_classes);
cluster_index(:,1) = cluster_index(:,1)-1;

start_index1 = 1;
start_index2 = start_index1 + n_class1;
start_index3 = start_index2 + n_class2;
get_sequenced_data(cluster_index, start_index1, n_class1, class1_seq_lens, 1);
get_sequenced_data(cluster_index, start_index2, n_class2, class2_seq_lens, 2);
get_sequenced_data(cluster_index, start_index3, n_class3, class3_seq_lens, 3);



function idx = quantize(class_data)
    [idx, means] = kmeans(class_data, 20);
end



function [class_data, seq_lens] = get_class_data(filename)
    class_data = [];
    seq_lens = [];
    fileID = fopen(filename); 
    while ~feof(fileID)
        %for one sequence
        C = [];
        P = [];
        C = textscan(fileID,'%d \n %s \n %d \n');
        seq_len = C{3};
        P = textscan(fileID, '%f', 2*seq_len);
        for i = 1:2:2*seq_len
            class_data =  [class_data; P{1}(i) P{1}(i+1)];
        end
        seq_lens = [seq_lens seq_len];
    end
    fclose(fileID);
end

function get_sequenced_data(cluster_index, start_index, n_class, class_seq_lens, class_no)
    class_sequence = [];
    sample_ctr = 0;
    
    s_train = strcat('train_class_seq', num2str(class_no));
    s_train = strcat(s_train, '.txt');
    
    s_val = strcat('val_class_seq', num2str(class_no));
    s_val = strcat(s_val, '.txt');
    
    s_test = strcat('test_class_seq', num2str(class_no));
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
         if (sample_ctr <= 70)
            fprintf(fileID_train, '%d ', sequence);
            fprintf(fileID_train, '\n');
        end
        
        if (sample_ctr > 70 && sample_ctr < 90)
            fprintf(fileID_val, '%d ', sequence);
            fprintf(fileID_val, '\n');   
        end
        
        if (sample_ctr >= 90)
            fprintf(fileID_test, '%d ', sequence);
            fprintf(fileID_test, '\n');   
        end
        start_index_single_seq = start_index_single_seq+seq_len;
    end
    fclose(fileID_train);
    fclose(fileID_val);
    fclose(fileID_test);
    
end