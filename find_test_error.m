function [test_error] = find_test_error(w, X, y)

% find_test_error: compute the test error of a linear classifier w. The
%  hypothesis is assumed to be of the form sign([1 x(n,:)] * w)
%  Inputs:
%		w: weight vector
%       X: data matrix (without an initial column of 1s)
%       y: data labels (plus or minus 1)
%     
%  Outputs:
%        test_error: binary error of w on the data set (X, y) error; 
%        this should be between 0 and 1. 
y_rows = size(y,1);
X = horzcat(ones(size(X,1),1),X);
hypo_arr = X*w';
bin_err_arr = zeros(y_rows,1);
for i = 1:1:y_rows
    bin_err_arr = (sign(y(i)) == sign(hypo_arr));
end
test_error = mean(bin_err_arr);


