clear;
clc;
close all;
format compact;

load('multilinear-regression-example.mat');

s = 0.1;
I = 60;
alpha = 0.01;
lambda = 0.1;
degree = 3;

X = feature_scale(X);
[X_train, X_test, y_train, y_test] = split_data(X, y, s);

theta = zeros(number_poly_terms(degree, size(X_train, 2)), 1);
cost = zeros(I, 2);

for i = 1: I
    [J_train, dJ] = cost_function(degree, lambda, theta, X_train, y_train);
    [J_test, ~] = cost_function(degree, lambda, theta, X_test, y_test);
    
    cost(i, :) = [J_train, J_test];
    theta = theta - alpha .* dJ;
    
    if (mod(i, floor(0.1*I)) == 0)
        fprintf('Iteration: %d \t\t Train Cost: %.3e \t\t ', i, J_train);
        fprintf('Test Cost: %.3e\n', J_test);
    end
end

figure;
subplot(1, 2, 1);
plot(X, y, 'bo', X, hypothesis(degree, theta, X), 'r', 'LineWidth', 2);
xlabel('x');
ylabel('y');
axis xy;
axis square;

subplot(1, 2, 2);
plot(1: I, log(cost(:, 1)), 'b', 1: I, log(cost(:, 2)), 'LineWidth', 1);
xlabel('Number of Iterations');
ylabel('Log. Cost Function');
axis xy;
axis square;
legend('train', 'test');

clear dJ i J_test J_train X X_test X_train y y_test y_train