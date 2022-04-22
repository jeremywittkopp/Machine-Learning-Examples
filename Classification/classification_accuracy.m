function acc = classification_accuracy(degree, theta, x, y)
    h = hypothesis(degree, theta, x);
    h = h >= 0.5;
    acc = mean(y == h);
end
