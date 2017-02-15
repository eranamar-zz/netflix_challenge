# utils.R - collection of function used across all models including 
#           data processing, generic models and plotting functions
# Authors: Eran Amar & Tomer Levy
#
# NOTE:
#   gsubfn is not the CRAN version, but the github version. To install run
#     devtools::install_github("ggrothendieck/gsubfn")

library(ggplot2)
library(gplots)
library(gsubfn)
source('data_loader.R')

# -------- Loss calculations ----------

to_rank <- function(X, seps = c(-Inf, 1.5, 2.5, 3.5, 4.5, Inf)){
  # Converts real valued prediction to rank predictions by assigning
  # the values of X the relevant interval index in seps.
  # Exepcts X to be a vector
  
  X <- as.numeric(X)
  get.index <- Vectorize(function(x){
    which((x >= seps[1:length(seps)-1]) & (x < seps[2:length(seps)]))
  })
  return (get.index(X))
}

loss = function(pred, true_y) {
  # RMSE loss
  sqrt(mean((pred - true_y) ^ 2))
}

average.loss <- function(n, methods) {
  # Runs the methods in the names list `methods` n times over different random partitions 
  # and reports the losses scores in boxplot. Prints the means of the losses for each method.
  #
  # methods is a named list of functions with the signautre function(X, y, folds) expected
  # to return loss function.
  #
  # The function returs a named matrix of the loss values of each run.
  
  df <- c()
  for (k in  1:n) {
    load.data.global()
    list[X, y, folds] <-
      data_split(train.ratings, train.y.ratings, prob = c(0.6, 0.2, 0.2), keep_folds=T)
    res <- list()
    for (method in names(methods)) {
      res[[method]] <- methods[[method]](X, y, folds)
    }
    df <- rbind(df, res, deparse.level = 0)
  }
  boxplot(apply(df, 2, as.numeric))
  print(colMeans(apply(df, 2, as.numeric)))
  return(apply(df, 2, as.numeric))
}

colMode <- function(X, na.rm=F) {
  # Calculates mode (most common number) on the columns
  if (is.null(X)){
    return(NULL)
  }
  apply(X, 2, function(x){
    ux <- unique(x)
    if (na.rm){
      ux <- as.vector(na.omit(ux))
    }
    ux[which.max(tabulate(match(x, ux)))]
  })
}


# -------- Data Processing ----------

center_rating_per_person = function(rating) {
  rating - rowMeans(rating, na.rm = T)
}

center_rating_per_movie = function(rating) {
  sweep(rating, 2, colMeans(rating, na.rm = T), "-")
}

rowvalue.to.matrix <- function(X, rows){
  # Creates a matrix composed of single valued rows (values denoted in rows)
  if(is.null(row)){
    return(NULL);
  }
  return(matrix(rep(rows, ncol(X)), ncol=ncol(X)))
}

colvalue.to.matrix <- function(X, col){
  # Creates a matrix composed of single valued columns (values denoted in col)
  if(is.null(col)){
    return(NULL);
  }
  return(t(matrix(rep(col, nrow(X)), ncol=nrow(X))))
}

fill.na <- function(data, value){
  # Takes a data and single value or matrix, and replace or na with the relevant values
  if(is.matrix(value)){
    data[is.na(data)] <- value[is.na(data)]
  } else {
    data[is.na(data)] <- value
  }
  return(data)
}

# -------- Data Exploration  ----------

best_pair_correlation = function(corr) {
  # Finds and prints the pair with best correlation based
  # on correlation matrix.
  n = dim(corr)[1]
  highest.corr.indx = which.max(abs(corr - diag(rep(1, n))))
  i = highest.corr.indx / n
  i = (i - floor(i)) * n
  j = ceiling(highest.corr.indx / n)
  first = colnames(corr)[i]
  second = colnames(corr)[j]
  print(
    paste0(
      'strongest correlation for: \'',
      first,
      '\' and \'',
      second,
      '\' with corr: ',
      corr[i, j]
    )
  )
  return(c(first, second))
}

calculate_rating_stat = function(ratingMat) {
  # Reports intersting statistics about the movies in the rating matrix.
  # Specifically average rating and count of NA in each movie.
  movies_stat = data.frame(colMeans(ratingMat, na.rm = T),
                           colSums(is.na(ratingMat)))
  colnames(movies_stat) = c('avg_rating', 'NA_count')
  return(movies_stat)
}

# -------- Generic Models ----------

ensemble <- function(X, y, k, FUN, predict.FUN, ...){
  # Chooses random features from the matrix X (70% of the features) and
  # trains a predictor. runs the process k times and sorts the predictors
  # according to training error.
  if (missing(predict.FUN)){
    predict.FUN = predict
  }
  
  ls <- list()
  best <- list(score = Inf, fields = NULL)
  for (i in 1:k){
    fields <- sample(c(T,F), ncol(X), replace=T, prob = c(0.7, 0.3))
    l <- FUN(X[,fields], y, ...)
    score <- loss(y, predict.FUN(l, X[,fields]))
    ls[[i]] <- list(model = l,fields = fields, score = score )
  }
  return(list(
      predictors = ls[order(sapply(ls, function(x)(x$score)))],
      predict = predict.FUN
         ))
}

ensemble.predict <- function(ens, X, k){
  # Predicts X using the output of the ensemble() function.
  # Uses the first k predictors (sorted according to training error)
  if (missing(k)){
    k <- length(ens)
  }
  
  pred <- rep(0, nrow(X))
  for (e in ens$predictors[1:k]){
    pred <- pred + ens$predict(e$model, X[,e$fields])
  }
  pred<-pred/k 
  return(to_rank(pred))
}

ensemble.plot <- function(ens, X, y, s, ...){
  # Plots validation loss by the parameter k
  losses <- sapply(s, function(k){
      loss(y, ensemble.predict(ens, X, k))
  })
  plot(s, losses, ...)
}

impute.iteratively = function(FUN, X, y, as_ordered, imput.args, predict.args, predict.FUN){
  # Going iteratively over the columns of X, using FUN with imput.args as arguments to impute 
  # missing values on each column based on previous as the variables, and the current column as the response.
  #
  # Finally, learns a predictor of y on the imputed matrix using predict.FUN with predict.args as 
  # arguments. By default, predict.FUN will use FUN. 
  #
  # Returns a model containing the predictors for each feature as well for the reponse variable.
  
  
  if(missing(predict.FUN)){
    predict.FUN <- FUN
  }
  imputers <- list()
  
  X.not.na <- ! is.na(X)
  i <- 1
  for (name in colnames(X)){
    # Impute X
    if (any(!X.not.na[,name])){
      
      x_impt_train <- X[X.not.na[,name], 1:(i-1)]
      y_impt_train <- X[X.not.na[,name], name]
      
      if(as_ordered){
        y_impt_train <- as.ordered(y_impt_train)
      }
      
      imputers[[name]] <- do.call(FUN, c(list(x=x_impt_train, y=y_impt_train), imput.args))
      X[!X.not.na[,name], name] <- to_rank(predict(imputers[[name]], X[!X.not.na[,name], 1:(i-1)]))
    } else {
      imputers[[name]] <- NULL
    }
    i <- i + 1 
  }
  
  # Learn y
  if (as_ordered){
    y <- as.ordered(y)
  }
  
  model.fit <- do.call(predict.FUN, c(list(x=X, y=y), predict.args))
  
  return(list(
    imputers = imputers,
    fit = model.fit,
    imputed.train = X,
    as_ordered = as_ordered
  ))
}

impute.data <- function(model.imputed, testX){
  # Use the impute model to impute values in a matrix
  X.na <- is.na(testX)
  i <- 1
  for (name in colnames(testX)){
    imputer <- model.imputed$imputers[[name]]
    if (!is.null(imputer)){
      stopifnot(i>1)
      testX[X.na[,name], name] <- to_rank(predict(imputer, testX[X.na[,name], 1:(i-1)]))
    }
    i <- i + 1
  }
  
  return(testX)
}

impute.and.predict = function (model.imputed, testX){
  # Use the impute model to predict y on not complete textX.
  return(to_rank(predict(model.imputed$fit, 
                         impute.data(model.imputed,testX)
                         )
                 )
         )
}

# -------- Plotting function ----------

plot_time_trand = function(rating,
                           dates_mat,
                           movie_index,
                           time_interval = 30,...) {
  # Plots the average rating of a movie by date according to time_interval 
  # (plots with and without confidence intervals)
  
  rates = rating[, movie_index]
  dates = dates_mat[, movie_index]
  
  dates = round((dates - min(dates, na.rm = T)) / time_interval)
  
  unique_dates <- unique(dates)
  unique_dates <- na.omit(unique_dates)
  avg_rate <- rep(0, length(unique_dates))
  o <- order(unique_dates)
  i <- 1
  raters.per.bin <- rep(0, length(unique_dates))
  na.count <- 0
  confidence.int <- matrix(rep(0, 2*length(unique_dates)), ncol=2)
  for (d in unique_dates) {
    date.data <- rates[dates == d]
    avg_rate[i] <- mean(date.data, na.rm = T)
    date.data.sd <- sd(date.data, na.rm = T)
    raters.per.bin[i] <- sum(!is.na(date.data))
    
    confidence.int[i,] <- avg_rate[i] + qnorm(c(0.05, 0.95))*date.data.sd/sqrt(raters.per.bin[i])
    
    if (sum(dates == d, na.rm = T) < time_interval & F) {
      avg_rate[i] <- NA
      na.count <- na.count+raters.per.bin[i]
      raters.per.bin[i] <- NA
    }
    i = i + 1
  }
  plot(
    unique_dates[o],
    avg_rate[o],
    xlab = paste('date bins with width=', time_interval),
    ylab = 'avg rating in that time bin',
    ...
  )
  plotCI(unique_dates[o], avg_rate[o], ui=confidence.int[o,2], li=confidence.int[o,1], xlab = paste('date bins with width=', time_interval), ylab = 'avg rating in that time bin')
  print(raters.per.bin[o])
  print(na.count)
  title(paste('Time trend for:', colnames(rating)[movie_index]))
}

time_trend_for_multiple_timebins = function (X, movie_ind){
  # Repeats plot_time_trand for different intervals
  for (interval in c(7,30,180)){
    plot_time_trand(X$rating, X$dates, movie_ind, interval)
  }
}
confusion.distribution <- function(y.pred, y.true){
  # Given real valued predictor y.pred, plots the density of the different classes
  # according to y.true.
  d <- c()
  for (i in sort(unique(y.true))){
    d <- rbind(d, data.frame(predicted=y.pred[y.true==i], true_rank=toString(i)))
  }
  print(ggplot(d, aes(predicted, fill=true_rank)) + geom_density(alpha = 0.2))
}