function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

k = 1:m;

for iter = 1:num_iters
    
    sum1 = (1/m) * sum(theta(1) + theta(2) .* X(k,2) - y(k));
    temp1 = theta(1) - alpha * sum1;
    
    sum2 = (1/m) * sum((theta(1) + theta(2) .* X(k,2) - y(k)) .* X(k,2));
    temp2 = theta(2) - alpha * sum2;
    
    theta(1) = temp1;
    theta(2) = temp2;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
