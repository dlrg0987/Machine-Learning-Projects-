function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
%theta
[zR,zC] = size(z);

for i = 1:zR
   g(i,1) = (1/(1+exp(-1.*z(i,1))));
   g(i,2) = (1/(1+exp(-1.*z(i,2))));
   g(i,3) = (1/(1+exp(-1.*z(i,3))));

  







% =============================================================

end
