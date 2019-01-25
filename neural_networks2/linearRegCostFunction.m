function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% theta size (2,1)
h = X*theta;

% Calculate cost J
J_unreg = 1/(2*m)*sum(sum((h-y).^2));
% Exclude theta 1
J = lambda/(2*m)* sum(sum(theta(2:end, :).^2))+J_unreg;

% Calculate gradient-> size (1,2)
% Exclude grad1
grad_unreg = 1/m* sum((h-y).*X);
grad = lambda/m* theta'+ grad_unreg;
grad(1,1)=grad_unreg(1,1);










% =========================================================================

grad = grad(:);
end


