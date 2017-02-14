# compositions.R - collection of function attempting to build a more complex models
# Authors: Eran Amar & Tomer Levy

library(glmnet)
library(randomForest)
source('knn.R')
source('utils.R')
source('data_loader.R')


idea1 <- function(X, y, alpha){
  # Iterative imputing using random forest and predicting using linear
  # regression (with cross validation for the best lambda).
  
  glmnet.args <- list(alpha=alpha)
  rf.args <- list(ntree=20, nodesize = 15)
  
  imputed.model <- impute.iteratively(randomForest, X$train, y$train, as_ordered = F, 
                                      imput.args = rf.args, predict.args = glmnet.args, predict.FUN=cv.glmnet)
  
  
  preds <- to_rank(impute.and.predict(imputed.model, X$valid))
  l <- loss(preds, y$valid)
  print(paste('loss for impute with glmnet and predict with RF: ', l))
  return(l)
}


idea2 <- function(X, y, alpha){
  # Using impute-tree-predict-linear scheme to gain weight vectors for the features,
  # then using the weight of features to calculate distances in KNN.
  
  glmnet.args <- list(alpha=alpha)
  rf.args <- list(ntree=20, nodesize = 10)
  
  num.predictors <- 10
  bootstrap.size <- 250
  
  print('imputing the data')
  imputed.model <- impute.iteratively(randomForest, X$train, y$train, as_ordered = F, 
                                      imput.args = rf.args, predict.args = glmnet.args, predict.FUN=cv.glmnet)
  
  imputed.X.valid <- impute.data(imputed.model, X$valid)
  imputed.X.test <- impute.data(imputed.model, X$test)
  
  beta_coeff <- imputed.model$fit$glmnet.fit$beta[,imputed.model$fit$glmnet.fit$lambda==imputed.model$fit$lambda.min]
  
  beta_coeff <- abs(beta_coeff)^2
  #beta_coeff[beta_coeff < 0.03] <- 0
  beta_coeff <- beta_coeff / norm(as.matrix(beta_coeff))
  plot(beta_coeff)
  
  print(paste('cross validation on complete X to find best k. num.predictors =', num.predictors, 'boot.size =', bootstrap.size))
  knn.losses <- knn.bagging.model_selection(num.predictors, bootstrap.size, imputed.model$imputed.train, y$train, 
                                           imputed.X.valid, y$valid, dist.l2)
  
  best.k <- which.min(knn.losses)
  print(paste('best k=', best.k))
  
  dist <- function(a,b){
    res <- (a-b)^2
    return(sqrt(res %*% beta_coeff))
  }
  
  print('predicting on test set')
  best.knn.model <- knn.bagging.fit(num.predictors, bootstrap.size, imputed.model$imputed.train,y$train, imputed.X.test, dist)
  preds <- to_rank(knn.bagging.predict(best.knn.model, best.k))
  l <- loss(preds, y$test)
  print(paste('loss for idea2: ', l))
  return(l)
}


main.composition <- function() {
  # Utility function to run the different methods of composition in order to compare 
  # them
  methods <- list()
  
  for (alpha in c(0, 0.5, 1)){
    methods[[paste0("idea1.alpha_", alpha)]] <- 
      function(X, y, folds){
        idea1(X, y, alpha)
      }
    methods[[paste0("idea2.alpha_", alpha)]] <- 
      function(X, y, folds){
        idea2(X, y, alpha)
      }
  }
  
  average.loss(10, methods)
}
