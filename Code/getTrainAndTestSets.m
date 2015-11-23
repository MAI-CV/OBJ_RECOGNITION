function [Xtrain, Xtest, Ytrain, Ytest] = getTrainAndTestSets(...
                                mnist, digit_t, digit_f,...
                                num_train, num_test)
    digit_t_filename = ['train' num2str(digit_t)];
    digit_f_filename = ['test' num2str(digit_f)];

    digit_t_total = mnist.(digit_t_filename);
    digit_f_total = mnist.(digit_f_filename);
    
    num_instances_t = size(digit_t_total,1);
    indexes_t = randperm(num_instances_t,num_train);
    
    Xtrain = double(digit_t_total(indexes_t,:));
    Ytrain = ones(num_train,1);
    
    total_values = 1:num_instances_t;
    idx=ismember(total_values,indexes_t);
    indexes_test = total_values(~idx);
    
    aux = double(digit_t_total(indexes_test,:));
    Xtest = aux(1:num_test,:);
    Ytest = ones(num_train,1);
    
    
    
    num_instances_f = size(digit_f_total,1);
    indexes_f = randperm(num_instances_f,num_train);
    
    Xtrain = [Xtrain; double(digit_f_total(indexes_f,:))];
    Ytrain = [Ytrain; ones(num_train,1)-2];
    
    total_values = 1:num_instances_f;
    idx=ismember(total_values,indexes_f);
    indexes_test = total_values(~idx);
    
    aux = double(digit_f_total(indexes_test,:));
    Xtest = [Xtest; aux(1:num_test,:)];
    Ytest = [Ytest; ones(num_train,1)-2];
    
    Xtrain = Xtrain';
    Xtest = Xtest';
end