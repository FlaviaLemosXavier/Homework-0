#the GroupLens research lab generated their own database with over 20 million ratings for over 27,000 movies by more than 138,000 users. We make a small subset of this data available via the dslabs package:

library(tidyverse)
library(dslabs)
data("movielens")

movielens %>% as_tibble()


#Each row represents a rating given by one user to one movie
ncol(movielens)
names(movielens)
dim(movielens)

#We can see the number of unique users that provided ratings and how many unique movies were rated:
movielens %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#If we multiply those two numbers, we get a number larger than 5 million, yet our data table has about 100,000 rows. This implies that not every user rated every movie. So we can think of these data as a very large matrix, with users on the rows and movies on the columns, with many empty cells.
# users on the rows and movies on the columns, with many empty cells
# The gather function permits us to convert it to this format, but if we try it for the entire matrix, it will crash R
#Let's show the matrix for seven users and five movies.

keep <- movielens %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)

#table with top_n 5 movies and 7 userID, from 13 to 20 
tab <- movielens %>%
  filter(userId %in% c(13:20)) %>% 
  filter(movieId %in% keep) %>% 
  select(userId, title, rating) %>% 
  spread(title, rating)
tab %>% knitr::kable()


# sample
users <- sample(unique(movielens$userId), 100)

#distribution of the predictors of the sample: movies and users

movielens %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")



#You can think of the task of a recommendation system as filling in the NAs in the table above. To see how sparse the matrix is, here is the matrix for a random sample of 100 movies and 100 users with yellow indicating a user/movie combination for which we have a rating.

#This machine learning challenge is more complicated than what we have studied up to now because each
#outcome Y has a different set of predictors. To see this, note that if we are predicting the rating for movie
#i by user u, in principle, all other ratings related to movie i and by user u may used as predictors, but
#different users rate different movies and a different number of movies. Furthermore, we may be able to use
#information from other movies that we have determined are similar to movie i or from users determined to
#be similar to user u. In essence, the entire matrix can be used as predictors for each cell.

#Let's look at some of the general properties of the data to better understand the challenges.
#The first thing we notice is that some movies get rated more than others. Here is the distribution:

movielens %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

#This should not surprise us given that there are blockbuster movies watched by millions and artsy, independent movies watched by just a few.
#Our second observation is that some users are more active than others at rating movies:

movielens %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

#To see how this is a type of machine learning, notice that we need to build an algorithm with data we have collected that will then be applied outside our control, as users look for movie recommendations. 
# So let's create a test set to assess the accuracy of the models we implement.

library(caret)
set.seed(755)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.2, list = FALSE)
train_set <- movielens[-test_index,]
test_set <- movielens[test_index,]

#To make sure we don't include users and movies in the test set that do not appear in the training set, we remove these entries using the semi_join function:
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


#Loss function

## To compare different models or to see how well we're doing compared to some
# baseline, we need to quantify what it means to do well. We need a loss
# function. The Netflix challenge used the typical error and thus decided on a
# winner based on the residual mean squared error on a test set.

# So if we define yui as the rating for movie i by user u and y hat ui as our
# prediction, then the residual mean squared error is defined as follows. Here n
# is a number of user movie combinations and the sum is occurring over all these
# combinations. Remember that we can interpret the residual mean squared error
# similar to standard deviation. It is the typical error we make when predicting
# a movie rating. If this number is much larger than one, we're typically
# missing by one or more stars rating which is not very good. So let's quickly
# write a function that computes this residual means squared error for a vector
# of ratings and their corresponding predictors. It's a simple function that
# looks like this.
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#movie i
# user u
#Yu,i rating for movie i by user u
#^Yi, u prediction
#Remember that we can interpret the residual mean squared error similar to standard deviation.
#It is the typical error we make when predicting a movie rating.

# And now we're ready to build models and compare them to each other.
#if RMSE>1, not a good prediction

#Building the Recommendation System

#The Netflix challenge winners implemented two general classes of models.
#One was similar to k-nearest neighbors, where you found movies that were similar to each other and users that were similar to each other.
#The other one was based on an approach called matrix factorization.
#That's the one we we're going to focus on here.
#So let's start building these models. Let's start by building the simplest possible recommendation system.
#We're going to predict the same rating for all movies, regardless of the user and movie.
#We start with a model that assumes the same rating for all movies and all users, with all the differences explained by random variation: If  ??  represents the true rating for all movies and users and  ??  represents independent errors sampled from the same distribution centered at zero, then: 
#Yu,i=??+??u,i
#?? <-true ratings
#??u,i<-represents independent errors sampled from the same distribution centered at zero,
#In this case, the least squares estimate of  ??  - the estimate that minimizes the root mean squared error - is the average rating of all movies across all users.
#We can improve our model by adding a term,  bi , that represents the average rating for movie  i :
#Yu,i=??+bi+??u,i
#bi  is the average of Yu,i minus the overall mean for each movie i.
#We can further improve our model by adding  bu , the user-specific effect:
#Yu,i=??+bi+bu+??u,i
#Note that because there are thousands of  b 's, the lm function will be very slow or cause R to crash, so we don't recommend using linear regression to calculate these effects.

#mu_hat<- average rating of all movies across all users.
mu_hat <- mean(train_set$rating)
mu_hat

#And then we compute the residual mean squared error on the test set data.

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

#[1] 1.04822, naive_rmse it is pretty big 

#That's what's supposed to happen, because we know that the average minimizes the 
#residual mean squared error when using this model. 
#And you can see it with this code.
#So we get a residual mean squared error of about 1.

predictions <- rep(2.5, nrow(test_set))
RMSE(test_set$rating, predictions)
#[1] 1.489453

#To win the grand prize of $1 million, a participating team
#had to get to a residual mean squared error of about 0.857.
# So we can definitely do better.

#Now because as we go along we will be comparing different approaches,
#we're going to create a table that's going to store the results that we
#obtain as we go along.

#We're going to call it RMSE results.

rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

rmse_results

#Let´s see how we can do better
#We know from experience that some movies are just
#generally rated higher than others.
#We can see this by simply making a plot of the average rating
#So our intuition that different movies are rated differently is confirmed by data.

#So we can augment our previous model by adding a term, b_i,
#to represent the average rating for movie i.

#In statistics, we usually call these b's, effects.
#ut in the Netflix challenge papers, they refer to them
#as "bias," thus the b in the notation.

#fit <- lm(rating ~ as.factor(userId), data = movielens) 
#However, note that because there are thousands of b's, each movie gets
#one parameter, one estimate.(do not run this code)

#However, in this particular situation, we
#know that the least squares estimate, b-hat_i, is just the average of y_u,i
#minus the overall mean for each movie, i.
#So we can compute them using this code.

mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#Note that we're going to drop the hat notation in the code to represent the estimates going forward, just to make the code cleaner.
#So this code completes the estimates for the b's.

#We can see that these estimates vary substantially, not surprisingly.
#Some movies are good.Other movies are bad.

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

#remember the overall avg = 3.5
#So a b i of 1.5 implies a perfect five-star rating, once
#^Yu,i =  ^mu + ^bi

#Now let's see how much our prediction improves once we predict using the model that we just fit.

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

#We can use this code and see that our residual mean squared error did drop a little bit.
#We already see an improvement.

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))

#show the RMSE = model_1_rmse in the rmse_results:
rmse_results %>% knitr::kable()

#Now can we make it better?
#How about users?
#To explore the data, let's compute the average rating for user, u,
#for those that have rated over 100 movies.

#we can make a histogram of those users like this:

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

#Note that there is substantial variability across users, as well.
#Some users are very cranky.
#And others love every movie they watch, while others are somewhere in the middle.

#This implies that a further improvement to our model may be something like this.
#Yu,i=??+bi+bu+??u,i
#bu = user-specific effect

#how to improve it? We could use lm, but we will not do it because is a big model
#It will probably crash our computer.
# lm(rating ~ as.factor(movieId) + as.factor(userId))

#Instead, we will compute our approximation
#by computing the overall mean, u-hat, the movie effects, b-hat_i,
#and then estimating the user effects, b_u-hat,
#by taking the average of the residuals obtained

user_avgs <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#Now let's see how much our prediction improves once we predict using the model that we just fit.
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

#We can use this code and see that our residual mean squared error did drop a little bit.
#We already see an improvement.

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()
