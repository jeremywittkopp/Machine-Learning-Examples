function h = hypothesis(degree, theta, x)
    x = poly_features(degree, x);
    g = @(z) 1 ./ (1 + exp(-z));
    h = g(x * theta);
end