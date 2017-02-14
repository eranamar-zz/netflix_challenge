# linear.R - collection of function to fit and predict linear regression based modules
# Authors: Eran Amar & Tomer Levy

source("utils.R")
source("data_loader.R")
source("movie_clustering.r")
library(glmnet)

try.groups <- function(X, y, folds, k){
  # Cluster user into k clusters based on cosine similarity.
  # Impute missing values in each cluster based on column mode.
  # Predict using linear regression.
  
  groups <- cluster.users(train.ratings, k, F)
  folds.groups <- list()
  for (set in names(X)){
    folds.groups[[set]] <- groups[folds == set]
  }
  
  groups.range <- min(groups):max(groups)
  calc.groups.mode <- function(data, groups){
    sapply(groups.range, function(g){
      if (all(groups != g)) {
        return(NA)
      }
      return(colMode(data[groups == g,], na.rm = T))
    })
  }
  
  modes <- list()
  for (set in names(X)){
    if (!is.null(X[[set]])){  
      set.mode <- colMode(X[[set]], na.rm = T)
      modes[[set]] <- calc.groups.mode(X[[set]], folds.groups[[set]]) 
      modes[[set]][is.na(modes[[set]])] <- rowvalue.to.matrix(modes[[set]], set.mode)[is.na(modes[[set]])]
    }
  }

  for (g in groups.range){
    for (set in names(X)){
      if (!is.null(X[[set]])){  
        na.g <- is.na(X[[set]])
        na.g[folds.groups[[set]] != g,] <- F
        X[[set]][na.g] <- colvalue.to.matrix(X[[set]], modes[[set]][,g])[na.g]
      }
    }
  }
  
  l <- glmnet(X$train, y$train, alpha = 0, lambda=c(0))
  return(loss(y$valid, to_rank(predict(l, X$valid))))
}


main.linear <- function() {
  # Utility function to run the different methods of linear regression in 
  # order to compare them
  methods <- list(
    iterative = 
      function(X, y, folds){
        print("Imputing using linear model (iteratively) - given order")
        model <- impute.iteratively(glmnet, X = X$train, y = y$train, as_ordered = F, 
                                    imput.args = list(alpha = 0, lambda = c(0)),
                                    predict.args = list(alpha = 0, lambda = c(0)))
        preds <- impute.and.predict(model, X$valid)
        return(loss(y$valid, to_rank(preds)))
      },
    
    iterative.rand = 
      function(X, y, folds){
        print("Imputing using linear model (iteratively) - random order")
        order <- c(1:14, sample(15:99))
        model <- impute.iteratively(glmnet, X = X$train[,order], y = y$train, as_ordered = F, 
                                    imput.args = list(alpha = 0, lambda = c(0)),
                                    predict.args = list(alpha = 0, lambda = c(0)))
        preds <- impute.and.predict(model, X$valid[,order])
        return(loss(y$valid, to_rank(preds)))
      },
    
    iterative.na.order =
      function(X, y, folds){
        print("Imputing using linear model (iteratively) - na order")
        order <- order(colSums(is.na(X$train)))
        model <- impute.iteratively(glmnet, X = X$train[,order], y = y$train, as_ordered = F, 
                                    imput.args = list(alpha = 0, lambda = c(0)),
                                    predict.args = list(alpha = 0, lambda = c(0)))
        preds <- impute.and.predict(model, X$valid[,order])
        return(loss(y$valid, to_rank(preds)))
      },
    
    vanilla.row = 
      function(X, y, folds){
        print("Predicting by imputing the row mean")
        rows <- matrix(rep(rowMeans(X$train, na.rm=T), ncol(X$train)), ncol=ncol(X$train))
        models <- glmnet(fill.na(X$train, rows), y$train, alpha = 0, lambda = c(0))
        rows <- matrix(rep(rowMeans(X$valid, na.rm=T), ncol(X$valid)), ncol=ncol(X$valid))
        preds <- predict(models, fill.na(X$valid, rows), exact=T)
        return(loss(y$valid, to_rank(preds)))
      },
    
    vanilla.col.mean = 
      function(X, y, folds){
        print("Predicting by imputing the col mean")
        cols <- t(matrix(rep(colMeans(X$train, na.rm=T), nrow(X$train)), ncol=nrow(X$train)))
        models <- glmnet(fill.na(X$train, cols), y$train, alpha = 0, lambda = c(0))
        cols <- t(matrix(rep(colMeans(X$valid, na.rm=T), nrow(X$valid)), ncol=nrow(X$valid)))
        preds <- predict(models, fill.na(X$valid, cols), exact=T)
        return(loss(y$valid, to_rank(preds)))
      },
    
    vanilla.col.mode = 
      function(X, y, folds){
        print("Predicting by imputing the col mode")
        cols <- t(matrix(rep(colMode(X$train, na.rm=T), nrow(X$train)), ncol=nrow(X$train)))
        models <- glmnet(fill.na(X$train, cols), y$train, alpha = 0, lambda = c(0))
        cols <- t(matrix(rep(colMode(X$valid, na.rm=T), nrow(X$valid)), ncol=nrow(X$valid)))
        preds <- predict(models, fill.na(X$valid, cols), exact=T)
        return(loss(y$valid, to_rank(preds)))
      },
    
    vanilla.mode =
      function(X, y, folds){
        print("Predicting by imputing the mode")
        models <- glmnet(fill.na(X$train, 3), y$train, alpha = 0, lambda = c(0))
        preds <- predict(models, fill.na(X$valid, 3), exact=T)
        return(loss(y$valid, to_rank(preds)))
      },
    
    groups =
      function(X, y, folds){
        print("Imputing bt mean in groups")
        return(try.groups(X, y, folds, 13))
      }
  )
  
  average.loss(100, methods)
}

