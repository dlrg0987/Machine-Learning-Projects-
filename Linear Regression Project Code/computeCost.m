function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

v = zeros(length(X),1); % create empty vector to store H(x) for each training example
H = (theta'.* X); % creating matrix containing theta*x for each variable x in matrix X for all  training examples

for i = 1:length(H)
    v(i) = (H(i,1) + H(i,2)); % sum elements of H in column 1 and 2 for all rows and append values into empty vector v

end 

J = (1/(2*m))*(sum(((v - y).^2)));



% =========================================================================

end
