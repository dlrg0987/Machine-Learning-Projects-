function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
size(Theta1)
size(Theta2)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

A_1 = [ones(m, 1) X]; %Apply bias unit to first layer of neural network inputs
Z_2 = A_1*Theta1'; %Compute Theta1 * A1 for hidden layer
A_2 = sigmoid(Z_2); % Compute the sigmoid function for each unit in hidden layer
A_2 = [ones(m, 1) A_2]; % Apply the bias unit to the hidden unit to compute output
Z_3 = A_2*Theta2'; % Compute Theta2 * A2 for Output layer
A_3 = sigmoid(Z_3); % Compute the sigmoid function for each unit in output layer
A_3_temp = A_3'; % transpose got have the training examples as columns and the individual values of each column equal to the hypothesis of each class/label.

%loop generating vectors in every row of y to represent the class labels in y in the form of a row vector with 1 in the column index of the same value as the value of that class for each row in y and 0 for other columns in each row that has a column index not equal to the value of that class.
y_temp = zeros(size(y,1),size(A_3_temp,1));
for i = 1:size(y,1)
    for j = 1:size(A_3_temp,1)
        if y(i) == j
            y_temp(i,j) = 1;
        elseif y(i) ~= j
            y_temp(i,j) = 0;
        end
    end
end
            
A_3_K = ((-1.*y_temp').*log(A_3_temp))-((1-y_temp').*log((1-A_3_temp))); %Computing logistic regression formula for each Class K in each training example.
A_3_k_sum = sum(A_3_K); % Summing the values of each class after applying logistic regression formula for each trainman example.
A_3_m_sum = sum((A_3_k_sum')); %summing over all training examples to form a scalar
J = (1/m)*A_3_m_sum; %multiplying scalar computed in previous line by 1/m to compute cost function J (unregularised).

%regularising cost function
Theta1_temp = sum(Theta1.^2); %Sum all elements of Theta1 squared in each column
Theta2_temp = sum(Theta2.^2); %Sum all elements of Theta2 squared in each column
Theta1_temp(1) = 0; %Set element of index 1 to zero to prevent bias term being regularised
Theta2_temp(1) = 0; %Set element of index 1 to zero to prevent bias term being regularised

Theta1_sum = sum(Theta1_temp'); %Sum all elements of Theta1 for each activation function in neural network (i.e every element in the row vector Theta1_temp)
Theta2_sum = sum(Theta2_temp'); %Sum all elements of Theta2 for each activation function in neural network (i.e every element in the row vector Theta2_temp)

Reg_parameter = (lambda/(2*m))*(Theta1_sum+Theta2_sum); %Compute the regularised portion of the logistic regression cost function.

J = (((1/m)*A_3_m_sum) + Reg_parameter); %multiplying scalar computed in previous line by 1/m to compute cost function J (regularised).




% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for t = 1:m

    a_1 = X(t,(1:end)); %Set first layer of training example t to a_1
    a_1_temp = a_1';
    a_1_temp = [1 ; a_1_temp]; %Apply bias unit to first layer of neural network inputs
    a_1 = a_1_temp;

    z_2 = Theta1 * a_1; %Compute Theta1 * a_1 for hidden layer 2
    a_2 = sigmoid(z_2); % Compute the sigmoid function for each unit in hidden layer
    a_2 = [1 ; a_2];%Apply bias unit to first layer of neural network inputs

    z_3 = Theta2 * a_2; % Compute Theta2 * a_2 for Output layer
    a_3 = sigmoid(z_3); % Compute the sigmoid function for each unit in output layer

    %y_t_vec = zeros(size(a_3)); % generate zero vector with size equal to a_3 to hold values of 1 and 0 indicating which class a-3 belongs to in each index.
    %[value,index] = max(a_3); 
    %y_t_vec(index) = 1;

    y_t_vec = ([1:num_labels]==y(t))';

    delta_3 = a_3 - y_t_vec;

    delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z_2)];
    delta_2 = delta_2(2:end);

    % Big delta update
    Theta1_grad = Theta1_grad + delta_2 * a_1';
    Theta2_grad = Theta2_grad + delta_3 * a_2';
end

% Un-regularised gradients
%Theta1_grad = (1/m) * Theta1_grad;
%Theta2_grad = (1/m) * Theta2_grad;



    

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Regularised gradients
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
%
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
    
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
