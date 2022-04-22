function h = hypothesis(degree, theta, x)
    x = poly_features(degree, x);
    h = x * theta;
end