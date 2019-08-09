#Matrix Factorization
#Our earlier models fail to account for an important source of variation related to the fact that groups of movies and groups of users have similar rating patterns. We can observe these patterns by studying the residuals and converting our data into a matrix where each user gets a row and each movie gets a column: 
#ru,i=yu,i???b^i???b^u,
#where  yu,i  is the entry in row  u  and column  i .
#We can factorize the matrix of residuals  r  into a vector  p  and vector  q ,  ru,i???puqi , allowing us to explain more of the variance using a model like this:
#Yu,i=??+bi+bu+puqi+??i,j
#Because our example is more complicated, we can use two factors to explain the structure and two sets of coefficients to describe users:
#Yu,i=??+bi+bu+pu,1q1,i+pu,2q2,i+??i,j
#To estimate factors using our data instead of constructing them ourselves, we can use principal component analysis (PCA) or singular value decomposition (SVD).

train_small <- movielens %>% 
  group_by(movieId) %>%
  filter(n() >= 50 | movieId == 3252) %>% ungroup() %>% #3252 is Scent of a Woman used in example
  group_by(userId) %>%
  filter(n() >= 50) %>% ungroup()

y <- train_small %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()

#To facilitate our analysis we add row names =y and column names=movie names
rownames(y)<- y[,1]
y <- y[,-1]
colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])

#And we convert these residuals by removing the column and row averages.
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))
y <- sweep(y, 2, colMeans(y, na.rm=TRUE))

#If the model we've been using describes all the signal
#and the extra ones are just noise, then the residuals for different movies
#should be independent of each other.
#But they are not. Let´s see an example: they are very correlated

m_1 <- "Godfather, The"
m_2 <- "Godfather: Part II, The"
qplot(y[ ,m_1], y[,m_2], xlab = m_1, ylab = m_2)

#The same is true for The Godfather and Goodfellas.
m_1 <- "Godfather, The"
m_3 <- "Goodfellas"
qplot(y[ ,m_1], y[,m_3], xlab = m_1, ylab = m_3)

#We see a correlation between other movies as well.

m_4 <- "You've Got Mail" 
m_5 <- "Sleepless in Seattle" 
qplot(y[ ,m_4], y[,m_5], xlab = m_4, ylab = m_5)

#We can see a pattern.
cor(y[, c(m_1, m_2, m_3, m_4, m_5)], use="pairwise.complete") %>% 
  knitr::kable()

#we can see that there's a positive correlation between the gangster movies
#Godfathers and Goodfellas, and then there's
#a positive correlation between the romantic comedies You've
#Got Mail and Sleepless in Seattle.

#We also see a negative correlation between the gangster movies and the romantic comedies.
#This means that users that like gangster movies a lot tend to not like romantic comedies and vise versa.

#This result tells us that there is structure in the data that the model does not account for.
#So how do we model this?
#Here is where we use matrix factorization.
#we are going to define factors
#Here's an illustration of how we could use some structure
#to predict the residuals.
#This structure could be explained using the following coefficients.
#We assign a 1 to the gangster movies and a minus one to the romantic comedies.
set.seed(1)
options(digits = 2)
Q <- matrix(c(1 , 1, 1, -1, -1), ncol=1)
rownames(Q) <- c(m_1, m_2, m_3, m_4, m_5)
P <- matrix(rep(c(2,0,-2), c(3,5,4)), ncol=1)
rownames(P) <- 1:nrow(P)

X <- jitter(P%*%t(Q))
X %>% knitr::kable(align = "c")

cor(X)

t(Q) %>% knitr::kable(aling="c")
P
#Note that we can also reduce the users to three groups, those
#that like gangster movies but hate romantic comedies,
#the reverse, and those that don't care.

#The main point here is that we can reconstruct
#this data that has 60 values with a couple of vectors totaling 17 values.
#Those two vectors we just showed can be used to form the matrix with 60 values.
#We can model the 60 residuals with the 17 parameter model like this.

#  q,ru,i??? puqi

#And this is where the factorization name comes in.
#We have a matrix r and we factorized it into two things, the vector p,
#and the vector q.
#actually we can factorize more, eg between the user that like Al Pacino Movies and that does not like:
set.seed(1)
options(digits = 2)
m_6 <- "Scent of a Woman"
Q <- cbind(c(1 , 1, 1, -1, -1, -1), 
           c(1 , 1, -1, -1, -1, 1))
rownames(Q) <- c(m_1, m_2, m_3, m_4, m_5, m_6)
P <- cbind(rep(c(2,0,-2), c(3,5,4)), 
           c(-1,1,1,0,0,1,1,1,0,-1,-1,-1))/2
rownames(P) <- 1:nrow(X)

X <- jitter(P%*%t(Q), factor=1)
X %>% knitr::kable(align = "c")

cor(X)

t(Q) %>% knitr::kable(aling="c")

P



six_movies <- c(m_1, m_2, m_3, m_4, m_5, m_6)
tmp <- y[,six_movies]
cor(tmp, use="pairwise.complete")

#So now we have to figure out how to estimate factors from the data as
#opposed to defining them ourselves.
#One way to do this is to fit models, but we can also
#use principle component analysis or equivalently, the singular value
#decomposition to estimate factors from data.