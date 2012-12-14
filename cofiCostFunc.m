function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unroll the X and Theta matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% Initialize the return values            
X_grad = zeros(size(X));        % partial derivative gradient for X
Theta_grad = zeros(size(Theta));% partial derivative gradient for Theta

% Calculate the cost, which is the size of prediction errors
predictions = X*Theta';
errors = (predictions - Y).*R;
J = (1/2) * sum(sum(errors.^2));

% Add regularization to penalize overfitting
reg_Theta = (lambda/2) * sum(sum(Theta.^2));
reg_X = (lambda/2) * sum(sum(X.^2));
J = J + reg_Theta + reg_X;

% Calculate partial derivative gradients
X_grad = errors*Theta;
Theta_grad = errors'*X;

% Add regularization terms
X_grad = X_grad + lambda*X;
Theta_grad = Theta_grad + lambda*Theta;

% Roll up the parameters
grad = [X_grad(:); Theta_grad(:)];

end