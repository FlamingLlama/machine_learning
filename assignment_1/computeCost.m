function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

k = 1:m;

J = (1 / (2*m)) * sum(((theta(1) + theta(2) .* X(k,2)) - y(k)).^2);

end
