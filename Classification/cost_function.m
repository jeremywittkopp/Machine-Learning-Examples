function [J, dJ] = cost_function(degree, lambda, theta, x, y)
    f = ones(length(theta), 1);
    f(1) = 0;
    
    h = hypothesis(degree, theta, x);
    J = -mean(y .* log(h) + (1 - y) .* log(1 - h)) + lambda * sum(f .* ...
        theta.^2) ./ (2 * size(x, 1));
    
    x = poly_features(degree, x);
    dJ = x' * (h - y) ./ size(x, 1) + f .* theta ./ size(x, 1);
end
