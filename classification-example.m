clear;
clc;
close all;
format compact;

load('multiclass-classification-example.mat');

if (max(y) == 1)
    y = y + 1;
end

s = 0.1;
I = 100;
alpha = 0.1;
lambda = 0;
degree = 2;

X = feature_scale(X);
[X_train, X_test, y_train, y_test] = split_data(X, y, s);

theta = zeros(number_poly_terms(degree, size(X_train, 2)), max(y));
cost = zeros(I, max(y), 2);
acc = zeros(I, max(y), 2);

for c = 1: max(y)
    for i = 1: I
        [J_train, dJ] = cost_function(degree, lambda, theta(:, c), ...
            X_train, y_train == c);
        [J_test, ~] = cost_function(degree, lambda, theta(:, c), ...
            X_test, y_test == c);

        acc(i, c, 1) = classification_accuracy(degree, theta(:, c), ...
            X_train, y_train == c);
        acc(i, c, 2) = classification_accuracy(degree, theta(:, c), ...
            X_test, y_test == c);

        cost(i, c, :) = [J_train, J_test];
        theta(:, c) = theta(:, c) - alpha .* dJ;

        if (mod(i, floor(0.1*I)) == 0)
            fprintf('Class: %d \t\t Iteration: %d \t\t ', c, i);
            fprintf('Train Cost: %.3f \t\t ', J_train);
            fprintf('Test Cost: %.3e\n', J_test);
        end
    end
end

cost = squeeze(mean(cost, 2));
acc = squeeze(mean(acc, 2));

figure;
subplot(1, 3, 1);
hold on;
for c = 1: max(y)
    plot(X(y == c, 1), X(y == c, 2), 'o');
end
hold off;
hold on;
for c = 1: max(y)
    decision_boundary(degree, 100, theta(:, c), X);
end
hold off;
xlabel('x_1');
ylabel('x_2');
axis xy;
axis square;

subplot(1, 3, 2);
plot(1: I, log(cost(:, 1)), 'b', 1: I, log(cost(:, 2)), 'LineWidth', 1);
xlabel('Number of Iterations');
ylabel('Log. Cost Function');
axis xy;
axis square;
legend('Train', 'Test');

subplot(1, 3, 3);
plot(1: I, acc(:, 1), 'b', 1: I, acc(:, 2), 'LineWidth', 1);
xlabel('Number of Iterations');
ylabel('Classification Accuracy');
axis xy;
axis square;
legend('Train', 'Test');

clear c dJ i J_test J_train Mtrain Mtest N X_test X_train Xpoly_test  
clear Xpoly_train y_test y_train