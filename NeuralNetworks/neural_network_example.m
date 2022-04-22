clear;
clc;
close all;
format compact;

% This is just a toy model. There are three sets of points drawn from three
% different normal distributions, each centered around different points but
% each with the same variance.
load('binary-classification-example.mat');

% I am too lazy to debug this for binary classification, so here I will
% just convert binary classification to multi-class classification with two
% classes.
if (max(y) == 1)
    y = y + 1;
end

% Establish the topology of the neural network and scale the input
% features. This will help the neural network learn more efficiently.
N = [size(X, 2), 20, max(y)];
X = feature_scale(X);

[X_train, X_test, y_train, y_test] = split_data(X, y, 0.1);

% Specify the learning rate and the number of iterations to train the
% neural network on the data. Also specify the L2-regularization parameter
% (if desired).
I = 100;
alpha = 0.1;
lambda = 0.1;

% Randomly initialize the weights of the neural network. Also initialize 
% the cost function evolution as a function of number of iterations and the
% classification accuracy as a function of number of iterations.
theta = initialize_weights(1, N);
cost = zeros(I, 2);
acc = zeros(I, 2);

% Train the neural network on the data. Each time, perform forward
% propagation to make a new set of predictions, determine how different the
% two distributions are via their cross-entropy and compute the
% classification accuracy, then backpropagation the errors in order to
% update the weights.
for i = 1: I
    theta = backward_propagation(alpha, lambda, theta, X_train, y_train);
    
    acc(i, 1) = mean(y_train == prediction(theta, X_train));
    acc(i, 2) = mean(y_test == prediction(theta, X_test));
    
    cost(i, 1) = cost_function(lambda, theta, X_train, y_train);
    cost(i, 2) = cost_function(lambda, theta, X_test, y_test);
    
    if (mod(i, 0.1*I) == 0)
        fprintf('Iteration :: %d \t\t Train Cost: %.3e', i, cost(i, 1));
        fprintf('\t\t Test Cost: %.3e\n', cost(i, 2));
    end
end

figure;
subplot(1, 2, 1);
plot(1: I, log(cost(:, 1)), 'b', 1: I, log(cost(:, 2)), 'LineWidth', 1);
xlabel('Number of Iterations');
ylabel('Log. Cross Entropy');
axis xy;
axis square;
legend('Train', 'Test');

subplot(1, 2, 2);
plot(1: I, acc(:, 1), 'b', 1: I, acc(:, 2), 'LineWidth', 1);
xlabel('Number of Iterations');
ylabel('Classification Accuracy');
axis xy;
axis square;
legend('Train', 'Test');

% Clear some room in memory
clear acci i