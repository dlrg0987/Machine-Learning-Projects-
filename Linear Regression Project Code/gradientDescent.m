function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
      V = zeros(length(X),1); % create empty vector to store H(x) for each training          example
      h = (theta'.* X); % creating matrix containing theta*x for each variable x in matrix  X for all  training examples

      for i = 1:length(h)
          V(i) = (h(i,1) + h(i,2)); % sum elements of H in column 1 and 2 for all rows and    append values into empty vector v

      end 
   
      theta = (theta - (alpha*(1/m).*(sum((V-y).*X)')));
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
