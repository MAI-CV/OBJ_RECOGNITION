mnist = load('mnist_all.mat');
Xp = random_selection(mnist.test0',100);
Xn = random_selection(mnist.test1',100);
X = double([Xp Xn]);
y = [ones(size(Xp,2),1); -ones(size(Xn,2),1)]';

C = [.0001 .1 1 2 4];
for i = 1:5
K = X'*X;
model = svm(K,y,C(i));
Xtest = X;

K_test = X(:,model.svind)'*Xtest;
y_test = model.alphay(model.svind)'*K_test + model.b;
subplot(5,1,i)
plot(y_test);
end
