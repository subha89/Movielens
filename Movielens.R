
#Download MovieLens Data


if(!require(tidyverse)) install.packages("tidyverse", repos ="http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip",dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],title = as.character(title),genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

#Data analysis of Movielens


str(movielens)

#The movielens dataset has more than 10 million ratings. Each rating comes with a userId, a movieId, the rating, a timestamp and information about the movie like title and genre.

hist(movielens$rating)

summary(movielens$rating)

movielens$year <- as.numeric(substr(as.character(movielens$title),nchar(as.character(movielens$title))-4,nchar(as.character(movielens$title))-1))

plot(table(movielens$year))
```
#More recent movies get more userratings. Movies earlier than 1930 get few ratings, whereas newer movies, especially in the 90s get far more ratings.

avg_ratings <- movielens %>% group_by(year) %>% summarise(avg_rating = mean(rating))
plot(avg_ratings)
```
#Movies from earlier decades have more volatile ratings, which can be explained by the lower frequence of movie ratings. 

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Learners will develop their algorithms on the edx set
# For grading, learners will run algorithm on validation set to generate ratings

validation <- validation %>% select(-rating)

#Exploratory Data Analysis on edx data set#

paste('The edx dataset has',nrow(edx),'rows and',ncol(edx),'columns.')


edx %>% summarize(n_movies = n_distinct(movieId))

#There are 10677 movies

edx %>% summarize(n_users = n_distinct(userId))

#There are 69878 users.

drama <- edx %>% filter(str_detect(genres,"Drama"))
comedy <- edx %>% filter(str_detect(genres,"Comedy"))
thriller <- edx %>% filter(str_detect(genres,"Thriller"))
romance <- edx %>% filter(str_detect(genres,"Romance"))

paste('Drama has',nrow(drama),'movies')
paste('Comedy has',nrow(comedy),'movies')
paste('Thriller has',nrow(thriller),'movies')
paste('Romance has',nrow(romance),'movies')


#Results#I have used penealized least squares machine Learning algorithm ti get the best possible rate.

##Choose Optimal Penalty Rate ‘Lambda’

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

lambdas <- seq(0, 5, 0.25)

rmses <- sapply(lambdas,function(l){
  
  #Calculate the mean of ratings from the edx training set
  mu <- mean(edx$rating)
  
  #Adjust mean by movie effect and penalize low number on ratings
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  #ajdust mean by user and movie effect and penalize low number of ratings
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  #predict ratings in the training set to derive optimal penalty value 'lambda'
  predicted_ratings <- 
    edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, edx$rating))
})


plot(lambdas, rmses)


lambda <- lambdas[which.min(rmses)]
paste('Optimal RMSE of',min(rmses),'is achieved with Lambda',lambda)


#The minimal RMSE of 0.856695227644159 is achieved with Lambda 0.5. So we will use the same for prediction.

#Apply Lamda on Validation set

lambda <- 0.5

pred_y_lse <- sapply(lambda,function(l){
  
  #Derive the mearn from the training set
  mu <- mean(edx$rating)
  
  #Calculate movie effect with optimal lambda
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  #Calculate user effect with optimal lambda
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  #Predict ratings on validation set
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred #validation
  
  return(predicted_ratings)
  
})

#Export Prediction

write.csv(validation %>% select(userId, movieId) %>% mutate(rating = pred_y_lse), "submission.csv", na = "" , row.names = FALSE)

#Conclusion
#Hence Penalized least squares machine learning algorithm provides the rating close to the original rating.
