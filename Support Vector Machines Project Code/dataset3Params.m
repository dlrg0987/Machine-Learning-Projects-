function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

results = eye(64,3); %generates matrix of 64 rows with 3 columns with column 1 being the test value for C, column 2 being the test value for sigma and column 3 being the error for those test values. 64 rows are used since there are 8 test values for both sigma and C which results in 64 error predictions (8*8 = 64)

errorRow = 0;% row number that is being computed. in nested for loop below

for C_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
    for sigma_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
        errorRow = errorRow + 1; 

        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test)); %Train model for inputs X, y and C_test with gaussian kernel computed in previously using gaussianKErnal function and include sigma_test within vernal computed.
        
        predictions = svmPredict(model, Xval); % Predict hypothesis for model that was trained using cross validation X values.
        
        prediction_error = mean(double(predictions ~= yval)); % determine the error of cross validation set. 

        results(errorRow,:) = [C_test, sigma_test, prediction_error]; % enter the values for the current test value of C and sigma with the value of the error in the row of the results matrix with the current value of errorRow.    
    end
end

sorted_results = sortrows(results, 3); % sort the rows of the results matrix in ascending order of the values in column 3 which are the error values

C = sorted_results(1,1); %value of C which gives the lowest error
sigma = sorted_results(1,2); %value of sigma which gives the lowest error





% =========================================================================

end
