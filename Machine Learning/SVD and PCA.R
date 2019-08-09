#SVD and PCA

#The matrix factorization decomposition that we showed in the previous video that looks something
#like this is very much related to singular value decomposition and PCA.

#singular value decomposition as an algorithm that finds the vectors p and q that
#permit us to write the matrix of residuals r with m rows and n columns in the following way.
#ru,i=pu,1q1,i+pu,2q2,i+...+pu,mqm,i,

#But with the added bonus that the variability of these terms
#is decreasing and also that the p's are uncorrelated to each other.
#This may permit us to see that with just a few terms,
#we can explain most of the variability.

#Let's see an example with our movie data.
#To compute the decomposition, will make all the NAs zero.

y[is.na(y)] <- 0
y <- sweep(y, 1, rowMeans(y))
pca <- prcomp(y)

#The vectors q are called the principal components
#and they are stored in this matrix.

dim(pca$rotation)

#While the p vectors, which are the user effects, are stored in this matrix.

dim(pca$x)

#The PCA function returns a component with the variability
#of each of the principal components and we can access it like this and plot it.

plot(pca$sdev)

#We can also see that just with a few of these principal components
#we already explain a large percent of the data.

var_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
plot(var_explained)

install.packages("ggrepel")
library(ggrepel)
pcs <- data.frame(pca$rotation, name = colnames(y))
pcs %>%  ggplot(aes(PC1, PC2)) + geom_point() + 
  geom_text_repel(aes(PC1, PC2, label=name),
                  data = filter(pcs, 
                                PC1 < -0.1 | PC1 > 0.1 | PC2 < -0.075 | PC2 > 0.1))

#The first principle component shows the difference
#between critically acclaimed movies on one side.
#Here are the one extreme of the principal component.Here we can see critically acclaimed movies on one side.


pcs %>% select(name, PC1) %>% arrange(PC1) %>% slice(1:10)

#and blockbosters on the other.
pcs %>% select(name, PC1) %>% arrange(desc(PC1)) %>% slice(1:10)
# We can also see that the second principle component also
#seems to capture structure in the data.
#If we look at one extreme of this principle component,
#we see artsy independent films such as Little Miss Sunshine, the Truman
#Show, and Slumdog Millionaire.
pcs %>% select(name, PC2) %>% arrange(PC2) %>% slice(1:10)

#When we look at the other extreme, we see
#what I would call nerd favorites, The Lord of the Rings,
#Star Wars, The Matrix.

pcs %>% select(name, PC2) %>% arrange(desc(PC2)) %>% slice(1:10)

#So using principal components analysis, we
#have shown that a matrix factorization approach can
# find important structure in our data.

#recommended lab to fit the models with missing data