function [Ynorm, Ymean] = normalizeRatings(Y, R)
%NORMALIZERATINGS Preprocess data by subtracting mean rating for every 
%movie (every row)
%   [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
%   has a rating of 0 on average, and returns the mean rating in Ymean.
%

[m, n] = size(Y);
Ymean = zeros(m, 1);
Ynorm = zeros(size(Y));
for i = 1:m
    % Ymean(i) = mean(Y(i, R(i, :))); % comment this out
    % add these lines immediately after:
    laplaceAlpha = 1;
    laplaceClasses = 5;
    N = sum(R(i,:) == 1);
    fewRatingPrior = (N/n);
    Ymean(i) = (sum(Y(i,R(i,:))) + laplaceAlpha) ...
        /(N + laplaceClasses) * fewRatingPrior;
    % end added lines
    Ynorm(i, R(i, :)) = Y(i, R(i, :)) - Ymean(i);
end

end