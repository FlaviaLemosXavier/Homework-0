# We will apply what we have learned in the course on the Modified National Institute of Standards and Technology database (MNIST) digits, a popular dataset used in machine learning competitions

library(dslabs)
mnist <- read_mnist()

#The data set includes two components, a training set and a test set. Let´s see:
names(mnist)

#Each of these components includes a matrix with features in the columns. You can access them using code like this.
dim(mnist$train$images)

#It also includes a vector with the classes as integers.

class(mnist$train$labels)
table(mnist$train$labels)

# sample 10k rows from training set, 1k rows from test set. we'll consider a subset of the data set.
set.seed(123)
index <- sample(nrow(mnist$train$images), 10000)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])

index <- sample(nrow(mnist$test$images), 1000)
x_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])


#Common preprocessing steps include:
#standardizing or transforming predictors and
#removing predictors that are not useful, are highly correlated with others, have very few non-unique values, or have close to zero variation. 

library(matrixStats)
sds <- colSds(x)
qplot(sds, bins = 256, color = I('black'))

#remove predictors with near zero variance
library(caret)
nzv <- nearZeroVar(x)
image(matrix(1:784 %in% nzv, 28, 28))

#see the remain predictors
col_index <- setdiff(1:ncol(x), nav)
length(col_index)

#Model Fitting for MNIST Data

#the caret package requires that we add column names to the feature matrices.
colnames(x) <- 1:ncol(mnist$train$images)
colnames(x_test) <- colnames(mnist$train$images)
#In general, it is a good idea to test out a small subset of the data first to get an idea of how long your code will take to run.
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(x[,col_index], y,
                   method = "knn", 
                   tuneGrid = data.frame(k = c(1,3,5,7)),
                   trControl = control)
ggplot(train_knn)

n <- 1000
b <- 2
index <- sample(nrow(x), n)
control <- trainControl(method = "cv", number = b, p = .9)
train_knn <- train(x[index ,col_index], y[index,],
                   method = "knn",
                   tuneGrid = data.frame(k = c(3,5,7)),
                   trControl = control)
fit_knn <- knn3(x[ ,col_index], y,  k = 5)

y_hat_knn <- predict(fit_knn,
                     x_test[, col_index],
                     type="class")
cm <- confusionMatrix(y_hat_knn, factor(y_test))
cm$overall["Accuracy"]

cm$byClass[,1:2]

library(Rborist)
control <- trainControl(method="cv", number = 5, p = 0.8)
grid <- expand.grid(minNode = c(1,5) , predFixed = c(10, 15, 25, 35, 50))
train_rf <-  train(x[, col_index], y,
                   method = "Rborist",
                   nTree = 50,
                   trControl = control,
                   tuneGrid = grid,
                   nSamp = 5000)
ggplot(train_rf)
train_rf$bestTune

fit_rf <- Rborist(x[, col_index], y,
                  nTree = 1000,
                  minNode = train_rf$bestTune$minNode,
                  predFixed = train_rf$bestTune$predFixed)

y_hat_rf <- factor(levels(y)[predict(fit_rf, x_test[ ,col_index])$yPred])
cm <- confusionMatrix(y_hat_rf, y_test)
cm$overall["Accuracy"]

rafalib::mypar(3,4)
for(i in 1:12){
  image(matrix(x_test[i,], 28, 28)[, 28:1], 
        main = paste("Our prediction:", y_hat_rf[i]),
        xaxt="n", yaxt="n")
}

#Variable importance
#The Rborist package does not currently support variable importance calculations, but the randomForest package does.
#An important part of data science is visualizing results to determine why we are failing.

library(randomForest)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])
rf <- randomForest(x, y,  ntree = 50)
imp <- importance(rf)
imp

image(matrix(imp, 28, 28))

p_max <- predict(fit_knn, x_test[,col_index])
p_max <- apply(p_max, 1, max)
ind  <- which(y_hat_knn != y_test)
ind <- ind[order(p_max[ind], decreasing = TRUE)]
rafalib::mypar(3,4)
for(i in ind[1:12]){
  image(matrix(x_test[i,], 28, 28)[, 28:1],
        main = paste0("Pr(",y_hat_knn[i],")=",round(p_max[i], 2),
                      " but is a ",y_test[i]),
        xaxt="n", yaxt="n")
}

p_max <- predict(fit_rf, x_test[,col_index])$census  
p_max <- p_max / rowSums(p_max)
p_max <- apply(p_max, 1, max)
ind  <- which(y_hat_rf != y_test)
ind <- ind[order(p_max[ind], decreasing = TRUE)]
rafalib::mypar(3,4)
for(i in ind[1:12]){
  image(matrix(x_test[i,], 28, 28)[, 28:1], 
        main = paste0("Pr(",y_hat_rf[i],")=",round(p_max[i], 2),
                      " but is a ",y_test[i]),
        xaxt="n", yaxt="n")
}

#Ensembles
#Ensembles combine multiple machine learning algorithms into one model to improve predictions.
p_rf <- predict(fit_rf, x_test[,col_index])$census
p_rf <- p_rf / rowSums(p_rf)
p_knn <- predict(fit_knn, x_test[,col_index])
p <- (p_rf + p_knn)/2
y_pred <- factor(apply(p, 1, which.max)-1)
confusionMatrix(y_pred, y_test)

#Q1
#Apply all of these models using train with all the default parameters. You may need to install some packages. Keep in mind that you will probably get some warnings. Also, it will probably take a while to train all of the models - be patient!
models <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", "gamLoess", "multinom", "qda", "rf", "adaboost")

#Run the following code to train the various models:
library(caret)
library(dslabs)
set.seed(1) # use `set.seed(1, sample.kind = "Rounding")` in R 3.6 or later
data("mnist_27")

fits <- lapply(models, function(model){ 
  print(model)
  train(y ~ ., method = model, data = mnist_27$train)
}) 

names(fits) <- models

#Q2

length(mnist_27$test$y)
length(models)

pred <- sapply(fits, function(object) 
  predict(object, newdata = mnist_27$test))
dim(pred)

#Q3
#Now compute accuracy for each model on the test set.
#Report the mean accuracy across all models.
acc <- colMeans(pred == mnist_27$test$y)
acc
mean(acc)

#Q4
#Next, build an ensemble prediction by majority vote and compute the accuracy of the ensemble.
votes <- rowMeans(pred == "7")
y_hat <- ifelse(votes > 0.5, "7", "2")
mean(y_hat == mnist_27$test$y)

#Q5
#In Q3, we computed the accuracy of each method on the test set and noticed that the individual accuracies varied.
#How many of the individual methods do better than the ensemble?
#Which individual methods perform better than the ensemble?
ind <- acc > mean(y_hat == mnist_27$test$y)
sum(ind)
models[ind]

#Q6
# remove the methods that do not perform well and re-do the ensemble
#However, we could use the minimum accuracy estimates obtained from cross validation with the training data for each model. Obtain these estimates and save them in an object. Report the mean of these training set accuracy estimates.
acc_hat <- sapply(fits, function(fit) min(fit$results$Accuracy))
mean(acc_hat)

#Q7
#Now let's only consider the methods with an estimated accuracy of greater than or equal to 0.8 when constructing the ensemble.
ind <- acc_hat >= 0.8
votes <- rowMeans(pred[,ind] == "7")
y_hat <- ifelse(votes>=0.5, 7, 2)
mean(y_hat == mnist_27$test$y)

#Q1
data("tissue_gene_expression")
dim(tissue_gene_expression$x)

pc <- prcomp(tissue_gene_expression$x)
data.frame(pc_1 = pc$x[,1], pc_2 = pc$x[,2], 
           tissue = tissue_gene_expression$y) %>%
  ggplot(aes(pc_1, pc_2, color = tissue)) +
  geom_point()

#Q2
avgs <- rowMeans(tissue_gene_expression$x)
data.frame(pc_1 = pc$x[,1], avg = avgs, 
           tissue = tissue_gene_expression$y) %>%
  ggplot(aes(avgs, pc_1, color = tissue)) +
  geom_point()
cor(avgs, pc$x[,1])

#Q3
x <- with(tissue_gene_expression, sweep(x, 1, rowMeans(x)))
pc <- prcomp(x)
data.frame(pc_1 = pc$x[,1], pc_2 = pc$x[,2], 
           tissue = tissue_gene_expression$y) %>%
  ggplot(aes(pc_1, pc_2, color = tissue)) +
  geom_point()

#Q4
#For the first 10 PCs, make a boxplot showing the values for each tissue.
for(i in 1:10){
  boxplot(pc$x[,i] ~ tissue_gene_expression$y, main = paste("PC", i))
}

#Q5
#Plot the percent variance explained by PC number. Hint: use the summary function.

#How many PCs are required to reach a cumulative percent variance explained greater than 50%?
plot(summary(pc)$importance[3,])

#Dimension reduction
#example
names(iris)
iris$Species

head(iris)
#Let's compute the distance between each observation and see in an image.
library(dslabs)
library(dplyr)
library(ggplot2)

x <- iris[,1:4] %>% as.matrix()
d <- dist(x)
image(as.matrix(d), col = rev(RColorBrewer::brewer.pal(9, "RdBu")))

#Our predictors here have four dimensions, but three are very correlated:
cor(x)

#Importance of the components. If we apply PCA, we should be able to approximate this distance with just two dimensions, compressing the highly correlated dimensions.

pca <- prcomp(x)
#Using the summary function we can see the variability explained by each PC:
summary(pca)

#The first two dimensions account for 97% of the variability. Thus we should be able to approximate the distance very well with two dimensions. 
#You can see from the weights that the first PC1 drives most the variability and it clearly separates the first third of samples (setosa) from the second two thirds (versicolor and virginica).

data.frame(pca$x[,1:2], Species=iris$Species) %>% 
  ggplot(aes(PC1,PC2, fill = Species))+
  geom_point(cex=3, pch=21) +
  coord_fixed(ratio = 1)
#We see that the first two dimensions preserve the distance:
d_approx <- dist(pca$x[, 1:2])
qplot(d, d_approx) + geom_abline(color="red")

#This example is more realistic than the first artificial example we used, since we showed how we can visualize the data using two dimensions when the data was four-dimensional.

#MNIST Example
#The written digits example 784 features. Is there any room for data reduction? Can we create simple machine learning algorithms with using fewer features?
#Let's load the data:
library(dslabs)
if(!exists("mnist")) mnist <- read_mnist()
head(mnist)
#Let's try PCA and explore the variance of the PCs.
col_means <- colMeans(mnist$test$images)
pca <- prcomp(mnist$train$images)

pc <- 1:ncol(mnist$test$images)
qplot(pc, pca$sdev)

#We can see that the first few PCs already explain a large percent of the variability:
summary(pca)$importance[,1:4] 

#And just by looking at the first two PCs we see information about the class. Here is a random sample of 2,000 digits:

data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2],
           label=factor(mnist$train$label)) %>%
  sample_n(2000) %>% 
  ggplot(aes(PC1, PC2, fill=label))+
  geom_point(cex=3, pch=21)

#Now let's apply the transformation we learned with the training data to the test data, reduce the dimension and run knn on just a small number of dimensions.
#We try 36 dimensions since this explains about 80% of the data
#First fit the model:

library(caret)
k <- 36
x_train <- pca$x[,1:k]
y <- factor(mnist$train$labels)
fit <- knn3(x_train, y)

#Now transform the test set:
x_test <- sweep(mnist$test$images, 2, col_means) %*% pca$rotation
x_test <- x_test[,1:k]

#And we are ready to predict and see how we do:
y_hat <- predict(fit, x_test, type = "class")
confusionMatrix(y_hat, factor(mnist$test$labels))$overall["Accuracy"]

#With just 36 dimensions we get an accuracy well above 0.95.

Q1

data("tissue_gene_expression")
dim(tissue_gene_expression$x)
names(tissue_gene_expression)
head(tissue_gene_expression$x)
head(tissue_gene_expression$y)

#the first row
tissue_gene_expression$x[1,] 

#number of predictors
length(tissue_gene_expression$x[1,])

#the first column
tissue_gene_expression$x[,1]

#number of observation by tissue
length(tissue_gene_expression$x[,1])

pc <- prcomp(tissue_gene_expression$x)
data.frame(pc_1 = pc$x[,1], pc_2 = pc$x[,2], 
           tissue = tissue_gene_expression$y) %>%
  ggplot(aes(pc_1, pc_2, color = tissue)) +
  geom_point()

# Which tissue is in a cluster by itself? ANSWER: Liver

#
# Q2

# The predictors for each observation are measured using the same device and
# experimental procedure. This introduces biases that can affect all the
# predictors from one observation. For each observation, compute the average
# across all predictors, and then plot this against the first PC with color
# representing tissue. Report the correlation.
avgs <- rowMeans(tissue_gene_expression$x)

data.frame(pc_1 = pc$x[,1], avg = avgs, 
           tissue = tissue_gene_expression$y) %>%
  ggplot(aes(avgs, pc_1, color = tissue)) +
  geom_point()

cor(avgs, pc$x[,1])

# What is the correlation?
# [1] 0.5969088

# Q3
#

# We see an association with the first PC and the observation averages. Redo the
# PCA but only after removing the center. Part of the code is provided for you.

x <- with(tissue_gene_expression, sweep(x, 1, rowMeans(x)))
pc <- prcomp(x)
data.frame(pc_1 = pc$x[,1], pc_2 = pc$x[,2], 
           tissue = tissue_gene_expression$y) %>%
  ggplot(aes(pc_1, pc_2, color = tissue)) +
  geom_point()

#Q4
#For the first 10 PCs, make a boxplot showing the values for each tissue.

for(i in 1:10){
  boxplot(pc$x[,i] ~ tissue_gene_expression$y, main = paste("PC", i))
}

for(i in 1:7){
  boxplot(pc$x[,i] ~ tissue_gene_expression$y, main = paste("PC", i))
}
#colon, placenta

#Q5

pca <- prcomp(x)
#Using the summary function we can see the variability explained by each PC:
summary(pca)
