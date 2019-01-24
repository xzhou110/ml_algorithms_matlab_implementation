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
% Theta1 size (25, 401)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% Theta2 size (10, 26)
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% X size (5000,400), z2 size(5000,25), a2 size(5000,25)
X = [ones(m,1) X];
z2 = X*Theta1';
a2 = sigmoid(z2);

% a2 size (5000,25), z3 size(5000,10), h3 size(5000,10)
a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2';
h3 = sigmoid(z3);

[values, h] = max(h3, [], 2);

% Convert y back from digits to binary matrix
% Number of array element in y. This is used as num_rows
% rows = numel(y);
% % Maximum value of y. This is used as columns
% columns = max (y);
% y_pred_matrix = zeros (rows, columns);
% 
% % sub2ind: subscript to index. find the index of places that have value
% y_pred_matrix(sub2ind(size(y_pred_matrix), 1:numel(y), y')) = 1;

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

J_unreg = 1/m*sum(sum(-y_matrix.* log(h3) - (1-y_matrix).*log(1-h3)));

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
%
% Theta1 size (25, 401)
% Theta2 size (10, 26)
% delta3 size (5000,10)
delta3 = h3-y_matrix;

% delta2 size (25,5000)
%  (10,25)'*(5000,10)'->(25,5000)
delta2= (Theta2(:, 2:end))'*delta3'.* (sigmoidGradient(z2))';

% (5000,10)' * (5000,26) -> (10,26)
sum_delta2 = delta3'*(a2);

% (25, 5000) * (5000,401) -> (25, 401)
sum_delta1 = delta2 *(X);

Theta2_grad_unreg = 1/m * sum_delta2;
Theta1_grad_unreg = 1/m * sum_delta1;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


J = J_unreg+lambda/(2*m)*(sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:,2:end).^2)));


Theta2_grad =Theta2_grad_unreg+lambda/m*Theta2;
Theta2_grad(:,1)=Theta2_grad_unreg(:,1);

Theta1_grad =Theta1_grad_unreg+lambda/m*Theta1;
Theta1_grad(:,1)=Theta1_grad_unreg(:,1);





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
