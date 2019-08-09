#Classification and Regression Trees (CART)
#A tree is basically a flow chart of yes or no questions. The general idea of the methods we are describing is to define an algorithm that uses data to create these trees with predictions at the ends, referred to as nodes.
When the outcome is continuous, we call the decision tree method a regression tree.
Regression and decision trees operate by predicting an outcome variable  Y  by partitioning the predictors.
The general idea here is to build a decision tree and, at end of each node, obtain a predictor  y^ . Mathematically, we are partitioning the predictor space into  J  non-overlapping regions,  R1  ,  R2 , ...,  RJ  and then for any predictor  x  that falls within region  Rj , estimate  f(x) with the average of the training observations  yi  for which the associated predictor  xi  in also in  Rj .To fit the regression tree model, we can use the rpart function in the rpart package.
Two common parameters used for partition decision are the complexity parameter (cp) and the minimum number of observations required in a partition before partitioning it further (minsplit in the rpart package). 
If we already have a tree and want to apply a higher cp value, we can use the prune function. We call this pruning a tree because we are snipping off partitions that do not meet a cp criterion. 
#Code
# Load data
library(tidyverse)
library(dslabs)
data("olive")
olive %>% as_tibble()
table(olive$region)
olive <- select(olive, -area)

# Predict region using KNN
library(caret)
fit <- train(region ~ .,  method = "knn", 
             tuneGrid = data.frame(k = seq(1, 15, 2)), 
             data = olive)
ggplot(fit)
# Plot distribution of each predictor stratified by region
olive %>% gather(fatty_acid, percentage, -region) %>%
  ggplot(aes(region, percentage, fill = region)) +
  geom_boxplot() +
  facet_wrap(~fatty_acid, scales = "free") +
  theme(axis.text.x = element_blank())
# plot values for eicosenoic and linoleic
p <- olive %>% 
  ggplot(aes(eicosenoic, linoleic, color = region)) + 
  geom_point()
p + geom_vline(xintercept = 0.065, lty = 2) + 
  geom_segment(x = -0.2, y = 10.54, xend = 0.065, yend = 10.54, color = "black", lty = 2)


# load data for regression tree
data("polls_2008")
qplot(day, margin, data = polls_2008)

library(rpart)
fit <- rpart(margin ~ ., data = polls_2008)

# visualize the splits 
plot(fit, margin = 0.1)
text(fit, cex = 0.75)
polls_2008 %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col="red")
# change parameters
fit <- rpart(margin ~ ., data = polls_2008, control = rpart.control(cp = 0, minsplit = 2))
polls_2008 %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col="red")

# use cross validation to choose the best cp
library(caret)
train_rpart <- train(margin ~ .,method = "rpart",tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),data = polls_2008)
ggplot(train_rpart)

# access the final model and plot it
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)
polls_2008 %>% 
  mutate(y_hat = predict(train_rpart)) %>% 
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col="red")

# prune the tree 
pruned_fit <- prune(fit, cp = 0.01)

#Classification (Decision) Trees
#Classification trees, or decision trees, are used in prediction problems where the outcome is categorical. 
Decision trees form predictions by calculating which class is the most common among the training set observations within the partition, rather than taking the average in each partition.
Two of the more popular metrics to choose the partitions are the Gini index and entropy.
#Pros: Classification trees are highly interpretable and easy to visualize.They can model human decision processes and don't require use of dummy predictors for categorical variables.
Cons: The approach via recursive partitioning can easily over-train and is therefore a bit harder to train than. Furthermore, in terms of accuracy, it is rarely the best performing method since it is not very flexible and is highly unstable to changes in training data. 
#Code
# fit a classification tree and plot it
train_rpart <- train(y ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = mnist_27$train)
plot(train_rpart)

# compute accuracy
confusionMatrix(predict(train_rpart, mnist_27$test), mnist_27$test$y)$overall["Accuracy"]

#Random Forests
#Random forests are a very popular machine learning approach that addresses the shortcomings of decision trees. The goal is to improve prediction performance and reduce instability by averaging multiple decision trees (a forest of trees constructed with randomness).
The general idea of random forests is to generate many predictors, each using regression or classification trees, and then forming a final prediction based on the average prediction of all these trees. To assure that the individual trees are not the same, we use the bootstrap to induce randomness. 
A disadvantage of random forests is that we lose interpretability.
An approach that helps with interpretability is to examine variable importance. To define variable importance we count how often a predictor is used in the individual trees. The caret package includes the function varImp that extracts variable importance from any model in which the calculation is implemented. 
#Code
install.packages("randomForest")
library(randomForest)
fit <- randomForest(margin~., data = polls_2008) 
plot(fit)

polls_2008 %>%
  mutate(y_hat = predict(fit, newdata = polls_2008)) %>% 
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_line(aes(day, y_hat), col="red")

data=mnist_27$train
train_rf <- randomForest(y ~ ., data=mnist_27$train)
confusionMatrix(predict(train_rf, mnist_27$test), mnist_27$test$y)$overall["Accuracy"]

# use cross validation to choose parameter
train_rf_2 <- train(y ~ .,
                    method = "Rborist",
                    tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
                    data = mnist_27$train)
confusionMatrix(predict(train_rf_2, mnist_27$test), mnist_27$test$y)$overall["Accuracy"]

#Q1
library(rpart)
n <- 1000
sigma <- 0.25
set.seed(1, sample.kind = "Rounding") 
x <- rnorm(n, 0, 1)
y <- 0.75 * x + rnorm(n, 0, sigma)
dat <- data.frame(x = x, y = y)

fit <- rpart(y ~ ., data = dat)

#Q2
plot(fit)
text(fit)

#Q3
dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col=2)

#Q4
library(randomForest)
fit <- randomForest(y ~ x, data = dat)
  dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col = 2)
  
#Q5
plot(fit)

#Q6
library(randomForest)
fit <- randomForest(y ~ x, data = dat, nodesize = 50, maxnodes = 25)
  dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col = 2)
  
#Caret Package
  The caret package helps provides a uniform interface and standardized syntax for the many different machine learning packages in R. Note that caret does not automatically install the packages needed.
 
   Caret package links
  http://topepo.github.io/caret/available-models.html
  
  http://topepo.github.io/caret/train-models-by-tag.html
  
  Code
  

  library(tidyverse)
  library(dslabs)
  data("mnist_27")
  
  library(caret)
  train_glm <- train(y ~ ., method = "glm", data = mnist_27$train)
  train_knn <- train(y ~ ., method = "knn", data = mnist_27$train)

  y_hat_glm <- predict(train_glm, mnist_27$test, type = "raw")
  y_hat_knn <- predict(train_knn, mnist_27$test, type = "raw")
  
  confusionMatrix(y_hat_glm, mnist_27$test$y)$overall[["Accuracy"]]
  confusionMatrix(y_hat_knn, mnist_27$test$y)$overall[["Accuracy"]]
  
  #Tuning Parameters with Caret
  #The train function automatically uses cross-validation to decide among a few default values of a tuning parameter. 
  The getModelInfo and modelLookup functions can be used to learn more about a model and the parameters that can be optimized.
  We can use the tunegrid parameter in the train function to select a grid of values to be compared.
  The trControl parameter and trainControl function can be used to change the way cross-validation is performed.
  Note that not all parameters in machine learning algorithms are tuned. We use the train function to only optimize parameters that are tunable.

  #Code
  getModelInfo("knn")
  modelLookup("knn")
  train_knn <- train(y ~ ., method = "knn", data = mnist_27$train)
  ggplot(train_knn, highlight = TRUE)
  train_knn <- train(y ~ ., method = "knn", 
                     data = mnist_27$train,
                     tuneGrid = data.frame(k = seq(9, 71, 2)))
  ggplot(train_knn, highlight = TRUE)  
  
  #best parameter
  train_knn$bestTune
  
  #best-performing model
  train_knn$finalModel  
  
  #calculate the accuracy with the best-performing model, based on the trainning data
  confusionMatrix(predict(train_knn, mnist_27$test, type = "raw"),
                  mnist_27$test$y)$overall["Accuracy"]
  #Sometimes we like to change the way we perform cross-validation.If we want to do this, we need to use a trainControl function.
  #example with 10 validation   samples
  control <- trainControl(method = "cv", number = 10, p = .9)
  train_knn_cv <- train(y ~ ., method = "knn", 
                        data = mnist_27$train,
                        tuneGrid = data.frame(k = seq(9, 71, 2)),
                        trControl = control)
  ggplot(train_knn_cv, highlight = TRUE)
  
  train_knn$results %>% 
    ggplot(aes(x = k, y = Accuracy)) +
    geom_line() +
    geom_point() +
    geom_errorbar(aes(x = k, 
                      ymin = Accuracy - AccuracySD,
                      ymax = Accuracy + AccuracySD))
  
  install.packages("gam")

  modelLookup("gamLoess")  
#expand our grid
grid<-expand.grid(span=seq(0.15,0.65, len=10), degree=1)

# now we use the default cross-validation control parameters, to train our model
#then, select the best-performing model
train_loess <- train(y ~ ., 
                     method = "gamLoess",
                     tuneGrid=grid,
                     data = mnist_27$train)
ggplot(train_loess, highlight = TRUE)

#let´s see the result (It performs similarly to knn, but it is smoother than what we get with knn.)
confusionMatrix(data = predict(train_loess, mnist_27$test), 
                reference = mnist_27$test$y)$overall["Accuracy"]

#Note that not all parameters in machine-learning algorithms are tuned.For example, in regression models or in LDA, we fit the best model using the squares estimates or maximum likelihood estimates. Those are not tuning parameters. We obtained those using least squares, or MLE, or some other optimization technique.
#So in k-nearest neighbors, the number of neighbors is a tuning parameter.In regression, the number of predictors that we include could be considered a parameter that's optimized. So in the caret package, in the train function, we only optimize parameters that are tunable.

#This is an important distinction to make when using the caret package-- knowing which parameters are optimized, and which ones are not.

#Q1
library(caret)
library(dslabs)
set.seed(1991)
data("tissue_gene_expression")

fit <- with(tissue_gene_expression, 
            train(x, y, method = "rpart",
                  tuneGrid = data.frame(cp = seq(0, 0.1, 0.01))))
fit
fit$bestTune
ggplot(fit)   

#Q2

#best-performing model
fit$finalModel  

#calculate the accuracy with the best-performing model, based on the trainning data

fit_rpart <- with(tissue_gene_expression, 
                  train(x, y, method = "rpart",
                        tuneGrid = data.frame(cp = seq(0, 0.10, 0.01)),
                        control = rpart.control(minsplit = 0)))
ggplot(fit_rpart)
confusionMatrix(fit_rpart)

#Q3

library(randomForest)
fit <- randomForest(y ~ x, data = dat)
dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col = 2)
plot(fit_rpart$finalModel)
text(fit_rpart$finalModel)

#Q4
set.seed(1991)
library(randomForest)
fit <- with(tissue_gene_expression, 
            train(x, y, method = "rf", 
                  nodesize = 1,
                  tuneGrid = data.frame(mtry = seq(50, 200, 25))))

ggplot(fit)

confusionMatrix(fit)
fit$bestTune

#Q5
imp <- varImp(fit)
imp

#Q6

plot(fit$finalModel)
text(fit$finalModel)

tree_terms <- as.character(unique(fit_rpart$finalModel$frame$var[!(fit_rpart$finalModel$frame$var == "<leaf>")]))

data("tissue_gene_expression")
set.seed(1991)

x <- tissue_gene_expression$x
y <- tissue_gene_expression$y

grid <- data.frame(mtry = seq(50, 200, 25)) 
fit_rf <- train(x, y, method = "rf", 
                tuneGrid = grid, nodesize = 1) 

imp <- varImp(fit_rf)
imp





