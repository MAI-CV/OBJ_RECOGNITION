mnist = load('mnist_all.mat');


digit_t = 4;
digit_f = 7;
num_train = 100;
num_test = 100;
[Xtrain, Xtest, Ytrain, Ytest] = ...
        getTrainAndTestSets(mnist, digit_t, digit_f, num_train, num_test);
    
Cs = [0.0001 0.001 0.01];
for i = 1:size(Cs,2)
    K = Xtrain'*Xtrain;
    model = svm(K,Ytrain',Cs(i));

    K_test = Xtrain(:,model.svind)'*Xtest;
    y_test = model.alphay(model.svind)'*K_test + model.b;
    subplot(1,size(Cs,2),i)
    plot(y_test);
end
