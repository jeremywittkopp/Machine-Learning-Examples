function decision_boundary(degree, res, theta, x)
    % This function will draw a decision boundary. NOTA BENE: this only
    % works for input data that originally has two features. It cannot
    % handle higher dimension data (yet...).
    % 
    % INPUTS:
    %       degree (1x1 int)            Maximum degree features to generate
    %                                   from the input features.
    %       res (1x1 int)               How many samples to take when
    %                                   defining the prediction contour.
    %       theta (Nx1 array)           Learned parameters.
    %       x (MxN array)               Original input matrix.
    
    % Sample the relevant region of interest - from the smallest sample in
    % either dimension to the largest.
    x1 = linspace(min(x(:, 1)), max(x(:, 1)), res);
    x2 = linspace(min(x(:, 2)), max(x(:, 2)), res);
    
    % Initialize the prediction contour to all zeros. Then, for each point
    % in the region of interest, compute the pre-activation function at
    % that point.
    Z = zeros(res, res);
    for i = 1: res
        for j = 1: res
            xp = poly_features(degree, [x1(j), x2(i)]);
            Z(i, j) = xp * theta;
        end
    end
    
    % Make meshgrids of the two axes, and plot the contour of the
    % pre-activation function, specifically the Z=0 contour, since this
    % corresponds to a probability of 0.5.
    [X1, X2] = meshgrid(x1, x2);
    contour(X1, X2, Z, [0, 0], 'color', 'k', 'LineWidth', 2);
    axis xy;
    axis square;
end