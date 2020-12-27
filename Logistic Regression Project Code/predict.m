function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

Z = zeros(length(X),1); % create empty vector to store H(x) for each training example
H = (theta'.* X); % creating matrix containing theta*x for each variable x in matrix X for all  training examples

for i = 1:length(Z)
    Z(i) = (H(i,1) + H(i,2) + H(i,3)); % sum elements of H in column 1, 2 and 3 for all rows and append values into empty vector v
end

G = zeros(length(X),1); %create empty vector to store sigmoid(Z) (i.e sigmoid(theta1*X1 + theta2*X2 ...)) for each training example

for i = 1:length(Z)
   G(i) = (1/(1+(exp(-1*Z(i))))) %Apply sigmoid function for each value of Z and then append that value into vector G
end

% for loop checking value of vector G for each element and then based on the value assigning corresponding element in vector p to either 1 or 0, 1 if element in G is >= 0.5 and 0 if element in G < 0.5

for i = 1:length(p)
    if G(i) >= 0.5
        p(i) = 1
    elseif G(i) < 0.5
        p(i) = 0
    end
end






% =========================================================================


end
