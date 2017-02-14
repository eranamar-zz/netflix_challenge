# data_loader.R - collection of function used across all models for loading 
#                 and very basic parsing of the data.
# Authors: Eran Amar & Tomer Levy

load_data = function() {
  # Loads all the data (ranting and dates, train and test).
  # Returns a list of tables.
  
  movies = read.table('movie_titles_only.txt', sep = '\n')[, 1]
  
  train_X = list(
    rating = read.table('train_ratings_all.dat.txt'),
    dates = read.table('train_dates_all.dat.txt')
  )
  train_y = list(
    rating = read.table('train_y_rating.dat.txt'),
    dates = read.table('train_y_dates.dat.txt')
  )
  test_X = list(
    rating = read.table('test_ratings_all.dat.txt'),
    dates = read.table('test_dates_all.dat.txt')
  )
  test_y = list(dates = read.table('test_y_dates.dat.txt'))
  
  colnames(train_X$rating) = movies
  colnames(train_X$dates) = movies
  colnames(test_X$rating) = movies
  colnames(test_X$dates) = movies
  
  train_X$rating[train_X$rating == 0] = NA
  test_X$rating[test_X$rating == 0] = NA
  
  return(list(
    train = list(X = train_X,
                 y = train_y),
    test = list(X = test_X,
                y = test_y)
  ))
}

load.data.global <- function(){
  # Loads only the train rating data and saves it into the global environment
  
  movie.title <- read.csv("movie_titles.txt", header = F, col.names=c("year", "title"))
  train.date <- read.table("train_dates_all.dat.txt", header=F, na.strings ="0", 
                           col.names=movie.title$title)
  train.ratings <- read.table("train_ratings_all.dat.txt", header=F, na.strings ="0",
                              col.names=movie.title$title)
  train.y.date <- read.table("train_y_dates.dat.txt", header=F, na.strings ="0")
  train.y.ratings <- read.table("train_y_rating.dat.txt", header=F, na.strings ="0")
  
  assign("train.ratings", train.ratings, envir = .GlobalEnv)
  assign("train.y.ratings", train.y.ratings[,1], envir = .GlobalEnv)
  assign("train.date", train.date, envir = .GlobalEnv)
  assign("train.y.date", train.y.date[,1], envir = .GlobalEnv)
}


data_split <- function(X, y, prob = c(0.6,0.0,0.4), keep_folds = F){
  # Splits the data randomaly into train/test/validation set based on probabilities.
  # Returns a list of X, y, each being an list of train/test/validation sets.
  
  if (nrow(X) != length(y)){
    abort("Length of X must be the same as y")
  }
  folds <- sample(c("train","test","valid"), nrow(X), replace=T, prob = prob)
  res <- list(X = split.data.frame(as.matrix(X), folds),
              y = split(y, folds))
  if (keep_folds){
    res[["folds"]] <- folds
  }
  return(res)
}