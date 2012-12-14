%% "Programming Collective Intelligence - Building Smart Web 2.0 Applications"
% by Toby Segaran (O'Reilly Media, ISBN-10: 0-596-52932-5)
%
% When Stanford University initially offered two online courses which later
% turned into Coursera and Udacity, I took Machine Learning class by Andrew
% Ng, It was an excellent class, one of the best I ever took. 
% 
% The class also included a module on Collaborative Filtering, and
% interestingly it used a very different approach from the Segaran's, and
% it took direct advantage of matrix computational capabilities because the
% class was taught in Octave. 
%
% In this example, I am using fminunc function from Optimization toolbox
% rather than fmincg fuction that Andew Ng used. 

% Initialization
clear all; clc ; close all;

%% Create the movie review dataset 
% I will use the same example used in the original book.

% There are 7 movie critics:
% 'Lisa Rose', 'Gene Seymour', 'Michael Phillips', ...
%    'Claudia Puig', 'Mick LaSalle', 'Jack Matthiews', 'Toby'
critics = {'Lisa', 'Gene', 'Michael', ...
   'Claudia', 'Mick', 'Jack', 'Toby'}';

% There are 6 movies:
% 'Lady in the Water', 'Snakes on a Plane', 'Just My Luck',...
%    'Superman Returns', 'The Night Listener', 'You, Me and Dupree'
movies = {'Lady', 'Snakes', 'JustMyLuck',...
   'Superman', 'NightListener', 'Dupree'}';

% Data is a matrix of ratings by those  movies (row) x critics (col)
Y = ...
[2.5 3.0 2.5 0   3.0 3.0 0  ; ...
 3.5 3.5 3.0 3.5 4.0 4.0 4.5; ...
 3.0 1.5 0   3.0 2.0 0   0  ; ...
 3.5 5.0 3.5 4.0 3.0 5.0 4.0; ...
 3.0 3.0 4.0 4.5 3.0 3.0 0  ; ...
 2.5 3.5 0   2.5 2.0 3.5 1 ];

disp('Sample data has been loaded into Workspace.')
reviews = dataset({Y,critics{:}},'ObsNames',movies)

% In this dataset, the ratings range between 1 to 5 and 0 indicates no 
% rating. I need a new matrix R which is a logical array that returns 1 
% if a given movie is rated by a given user or 0 if not. 
R = Y > 0;

% This is needed to filter out unrated movies so that we can calculate an
% average, for example. 
fprintf('Average rating for ''Lady in the Water'': %.1f out of 5\n\n', ...
    mean(Y(1, R(1, :))));

clear reviews

%% Recommendation as an optimization problem
% Toby Segaran's approach in the book was to use Euclidean distance or
% Pearson coefficient to calculate similarity scores. 
%
% Andrew Ng's approach turns this into an optimization problem. You need to
% come up with a model with an arbitrary parameters to preduct a user's
% rating for a give movie. Then compare the predicted ratings with the
% actual ratings and tune the parameters until you minimize the prediction
% errors. 
% 
%%% Predicting ratings - Content-based approach
% Let's assume we can predict a user's rating for a movie if we know the
% past movie ratings and relevant attributes of those movies, such as
% romantic, action, etc. This is a linear regression: 
% 
% |prediction = param0*attr0 + param1*attr1 + param2*attr2...|
%
% where *attri0* = 1 and therefore *param0* is just a intercept term.
%
% This arrangement makes it possible to group all parameters into matrix
% Theta and all attributes into matrix X. Then the lear regression is now
% simplified as:
%
% |P = X*Theta'|
% 
% If we know the feature matrix X, then we can estimate the parameter
% matrix Theta by choosingthe value of Theta that minimize prediction
% errors. 

% Create fake X and Theta with 3 features (attributes)
num_features = 3;
num_movies = size(movies,1);
num_users = size(critics,1);

% Randomly initialize Theta and X
disp('Randomly generating X and Theta...');
X = randn(num_movies,num_features);
Theta = randn(num_users,num_features);

% Predict ratings and show the prediction errors
P = X*Theta';
errors = (P-Y).*R; % filter out unrated movies

disp('Prediction errors from fake X and Theta.')
prediction_errors = dataset({errors,critics{:}},'ObsNames',movies)

clear P errors num_features num_movies num_users prediction_errors X Theta

%% Feature learning 
% The problem with the previous approach is that we need to know the values
% of X in order to estimate the values of Theta, but how do you get the
% values of X in the first place? You need to figure out the relevant
% attributes of the all the movies in advance. 
%
% You can also estimate the values of X if we know the values of Theta, but
% that means we need to know user preferences in advance. This is a chicken
% and egg problem. 
%
% The solution is to randomly initialize both X and Theta and derive
% through optimization the values of both matrix that together minimizes
% prediction errors. 

% Normalize Ratings so that each movie has a rating of 0 on average, and 
% returns the mean rating in Ymean.
[Ynorm, Ymean] = normalizeRatings(Y, R);

% Set cost function parameters
num_users = size(Ynorm, 2);
num_movies = size(Ynorm, 1);
num_features = 3;
lambda = 1; % regularization parameter to control over/under fitting

% Randomly initialize Theta and X again
disp('Randomly generating X and Theta again...');
X = randn(num_movies,num_features);
Theta = randn(num_users,num_features);
init_params = [X(:) ; Theta(:)]; % roll X and Theta into one vector

% Cost Function
J = cofiCostFunc(init_params, Ynorm, R, num_users, num_movies, ...
                                  num_features, lambda);
fprintf('Cost at current X and Theta: %f \n\n', J);

clear J 

%% Learning the parameters - the optimization approach
% Now that you have the cost function and initial parameters, you can feed
% them to a minimization solver fminunc:

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Run optimization
theta = fminunc(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                num_features, lambda)), ...
                init_params, options);

% Unroll the returned theta back into X and Theta
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

disp('Parameter learning completed.');

% Calling collaborative Filtering Cost Function
init_params = [X(:) ; Theta(:)];
J = cofiCostFunc(init_params, Ynorm, R, num_users, num_movies, ...
                                  num_features, lambda);
fprintf('Cost at optimized X and Theta: %f \n\n', J);

clear J init_params lambda num_features options theta

%% Using the learned model to make predictions
% Now that we have optimized X and Theta, we can use them to predict
% ratings and give recommendations. As in the book, we come to the same
% recommendations for Toby. Your results may vary due to random 
% initialization.

P = X * Theta';              % predicted ratings
P = bsxfun(@plus, P, Ymean); % add back the mean to restore full ratings

% Toby is at the end of critics vector
toby_predictions = min(P(:,end),5); % don't give more than 5 starts
[~, idx] = sort(toby_predictions, 'descend');

disp('Recommendations for Toby:');
for i=1:num_movies
    j = idx(i);
    if R(j,end) == 0
        fprintf('Predicted rating %.1f for movie %s\n', ...
            toby_predictions(j), movies{j});
    end
end

disp(' ');
disp('Results from the book:');
disp('#1 NightListener');
disp('#2 Lady');
disp('#3 JustMyLuck');
disp(' ');

clear P i idx j toby_predictions

%% Finding similar users
% We can also use optmized Theta to compare among users and discover who
% have similar tastes to one another based on the learned preferences. In
% this case I use Euclidean distance to rank the users by similarity.
% 
% My results are very close but not exactly the same as those in the book.
% Your results may vary due to random initialization.

similar_users = zeros(num_users);

for i=1:num_users
    distances = bsxfun(@minus, Theta, Theta(i,:));
    distances = sqrt(sum(distances.^2,2));
    similar_users(i,:) = distances';
end

% Whose tastes are similar to Toby's?
[~, idx] = sort(similar_users(end,:));
idx = idx(2:end);
disp('Users who share similar tastes with Toby:');
for j = 1:size(idx,2)
    k = idx(j);
    fprintf('Distance %.1f for user %s\n', ...
            similar_users(end,k), critics{k});
end

disp(' ');
disp('Results from the book:');
disp('#1 Lisa');
disp('#2 Mick');
disp('#3 Claudia');
disp(' ');

clear distances i idx j k num_users similar_users

%% Finding similar movies
% We can also use optmized X to compare movies and discover which movies
% are related to one another based on the learned features. Again I use 
% Euclidean distance to rank the users by similarity.
%
% My results are very close but not exactly the same as those in the book.
% Your results may vary due to random initialization.

similar_movies = zeros(num_movies);

for i=1:num_movies
    distances = bsxfun(@minus, X, X(i,:));
    distances = sqrt(sum(distances.^2,2));
    similar_movies(i,:) = distances';
end

% What's similar to "Superman"?
[~, idx] = sort(similar_movies(4,:));
idx = idx(2:end);
disp('Movies similar to "Superman":');
for j = 1:size(idx,2)
    k = idx(j);
    fprintf('Distance %.1f for movie %s\n', ...
            similar_movies(4,k), movies{k});
end

disp(' ');
disp('Results from the book:');
disp('#1 Dupree');
disp('#2 Lady');
disp('#3 Snakes');
disp('#4 NightListener');
disp('#5 JustMyLuck');
disp(' ');

clear distances i idx j k num_movies similar_movies

%% Closing
% There is more subtlety to this approach than was discussed so far. 
% 
% * Running optimization steps multiple times to get the best minimum
% * Selecting the number of features and lambda for regularization
% * Experiment with Pearson Coefficient instead of Euclidean distance
% * Try it with MovieLens dataset. See below for the initial setup
% 
% It just comes down to run parameter sweep to get the best possible
% values. 

%% Using the MovieLens Dataset  - Run the learning algorithm 
% You can try the MovieLens dataset included in the
% <https://github.com/toshiakit/recommendations earlier project> on the
% approach discussed here. Download the dataset into "ml-data_0" folder. 

% Load data from u.data and u.item as follows:
[Y, movies, R]= loadMovieLens('ml-data_0');

% Y, R must be (movies x users) matrices
Y = Y'; R = R';

% Transpose movie from a row to column vector
movies = movies';

% Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

% Set cost function parameters
num_users = size(Ynorm, 2);
num_movies = size(Ynorm, 1);
num_features = 100;
lambda = 1; % regularization parameter to control over/under fitting

% Randomly initialize Theta and X again
disp('Randomly generating X and Theta again...');
X = randn(num_movies,num_features);
Theta = randn(num_users,num_features);
init_params = [X(:) ; Theta(:)]; % roll X and Theta into one vector

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 100);

% % Run optimization - this part takes a long processing time.
% fprintf('\nRunning the solver... \n');
% tStart = tic;
% theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
%                 num_features, lambda)), ...
%                 init_params, options);
% tEnd = toc(tStart);
% fprintf('%d minutes and %f seconds\n',floor(tEnd/60),rem(tEnd,60));
% 
% % Unroll the returned theta back into X and Theta
% X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
% Theta = reshape(theta(num_movies*num_features+1:end), ...
%                 num_users, num_features);
% 
% fprintf('Parameter learning completed.\n');
% 
% % Calling collaborative Filtering Cost Function
% init_params = [X(:) ; Theta(:)];
% J = cofiCostFunc(init_params, Ynorm, R, num_users, num_movies, ...
%                                   num_features, lambda);
% fprintf('Cost at optimized X and Theta: %f \n', J);
%
% % Now that we have optimized X and Theta, we can use them to predict
% % ratings and give recommendations. Using User ID = 87 example from the
% % book...
% 
% P = X * Theta';              % predicted ratings
% P = bsxfun(@plus, P, Ymean); % add back the mean to restore full ratings
% 
% % Pick User ID = 87
% user87_predictions = min(P(:,87),5); % don't give more than 5 starts
% [~, idx] = sort(user87_predictions, 'descend');
% 
% fprintf('\nRecommendations for User ID = 87:\n');
% for i=1:num_movies
%     j = idx(i);
%     if R(j,end) == 0
%         fprintf('Predicted rating %.1f for movie %s\n', ...
%             user87_predictions(j), movies{j});
%     end
% end
