function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X]; %Apply bias unit to first layer of neural network inputs
Z_2 = X*Theta1'; %Compute Theta1 * A1 for hidden layer
A_2 = sigmoid(Z_2); % Compute the sigmoid function for each unit in hidden layer
A_2 = [ones(m, 1) A_2] % Apply the bias unit to the hidden unit to compute output
Z_3 = A_2*Theta2'% Compute Theta2 * A2 for Output layer
A_3 = sigmoid(Z_3)% Compute the sigmoid function for each unit in output layer
A_3_temp = A_3'% transpose Output matrix to determine most likely class for each training example

[x,ix] = max(A_3_temp); %Output the index of the class with the highest probability by outputting the max value in each column and outputting the index of that value in ix

% for loop inputting the most likely class based on the index value that was most likely into vector p for each train set of X.
for i = 1:length(ix)
    p(i) = ix(i);
end








% =========================================================================


end
