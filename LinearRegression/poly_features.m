function x_new = poly_features(degree, x)
    % This function will add polynomial features from zero up to and 
    % including the degree specified by the user. The algorithm used here
    % is the brute force method of computing the subset-sum problem. There
    % exist better algorithms to use, but for now, since we are only using
    % small numbers of features, this should suffice.
    %
    % INPUTS:
    %       degree (1x1 int)            Maximum degree features to generate
    %                                   from the input features.
    %       x (MxN array)               Original input matrix.
    %
    % OUTPUT:
    %       x_new (MxD array)           Modified input matrix which
    %                                   includes the newly created 
    %                                   polynomial features. The number of 
    %                                   columns can be found by summing all
    %                                   the possible index combinations.
    
    % Get the number of training examples and the original number of
    % features before adding the polynomial features.
    [M, N] = size(x);

    % Create a cell array consisting of repeated arrays of integers from
    % zero to the maximum degree polynomial temrs the user inputted.
    A = cell(1, N);
    for i = 1: N
        A{i} = 0: degree;
    end

    % Create N-dim grids of these A-arrays. This will correspond to all of
    % the possible combinations of indices before contraining them. Each
    % cell will correspond to a different input feature.
    B = cell(size(A));
    [B{:}] = ndgrid(A{:});

    % This will simply add all of the B-arrays together. This will
    % correspond to the total degree of each possible combination.
    power_sum = zeros(size(B{1}));
    for i = 1: length(B)
        power_sum = power_sum + B{i};
    end

    % Here we will apply the contraint: remove all those combinations whose
    % sum does not correspond to the given degree. These will be saved in
    % individual cells, where each cell corresponds to a different degree
    % of polynomial terms. This way we can also add those terms whose
    % degree is less than that the maximum inputted degree.
    powers = cell(1, degree+1);
    for n = 0: degree
        filter = (power_sum == n);
        powers_n = zeros(nchoosek(n+N-1, N-1), N);
        for j = 1: N
            powers_n(:, j) = B{j}(filter);
        end

        powers{n+1} = powers_n';
    end

    % Finally generate the new polynomial features and add them to the new
    % input matrix.
    powers = cell2mat(powers)';
    x_new = ones(M, size(powers, 1));
    for i = 1: size(powers, 1)
        for j = 1: N
            x_new(:, i) = x_new(:, i) .* x(:, j).^powers(i, j);
        end
    end
end