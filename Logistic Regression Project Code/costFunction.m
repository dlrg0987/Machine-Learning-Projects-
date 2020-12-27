function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

theta

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta

Z = zeros(length(X),1); % create empty vector to store H(x) for each training example
H = (theta'.* X); % creating matrix containing theta*x for each variable x in matrix X for all  training examples

for i = 1:length(Z)
    Z(i) = (H(i,1) + H(i,2) + H(i,3)); % sum elements of H in column 1, 2 and 3 for all rows and append values into empty vector v
end

G = zeros(length(X),1); %create empty vector to store sigmoid(Z) (i.e sigmoid(theta1*X1 + theta2*X2 ...)) for each training example

for i = 1:length(Z)
   G(i) = (1/(1+(exp(-1*Z(i))))) %Apply sigmoid function for each value of Z and then append that value into vector G
end

%H(x) for logistic regression = G for every training example i

J = (1/m)*(sum((-1*y).*log(G)-(1-y).*log((1-G)))); %Calculation of cost function
  

grad = (grad + ((1/m).*(sum((G-y).*X)'))); % obtain gradient of cost function J with respect to theta of each feature X
















% =============================================================

end
