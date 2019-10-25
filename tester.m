train_data = table2array(clevelandtrain);
test_data = table2array(clevelandtest);
X = train_data(:,1:size(train_data,2)-1);
X = zscore(X);
Y = train_data(:,size(train_data,2));
% ~~ Changing y's from 1 and 0s to 1 and -1s
for i = 1:1:size(Y,1)
    if Y(i) == 0
        Y(i) = -1;
    end
end
X_test = test_data(:,1:size(test_data,2)-1);
Y_test = test_data(:,size(test_data,2));
X_test = zscore(X_test);
for j = 1:1:size(Y_test,1)
    if Y_test(j) == 0
        Y_test(j) = -1;
    end
end
% ~~~~~~~~~~ actual test ~~~~~~~~~~~~~
import logistic_reg.*
import find_test_error.*
w_init = zeros(1,14);
max_its = Inf;
eta = .01;
[t,w,e_in] = logistic_reg(X,Y,w_init,max_its,eta);
bce_test = find_test_error(w,X_test,Y_test);
bce_training = find_test_error(w,X,Y);
t
w
e_in
bce_test
bce_training



