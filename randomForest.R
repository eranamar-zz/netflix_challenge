# randomForest.R - collection of function to fit and predict random forest based modules
# Authors: Eran Amar & Tomer Levy

source("utils.R")
source("data_loader.R")
library(randomForest)


resample <- function(size_per_rank, train_X, train_Y){
  # Upsampling and downsampling the train data to achieve fixed proportion for every l
  indices <- c()
  for (rnk in 1:5){
    filtered <- train_Y == rnk
    stopifnot(any(filtered))
    sample_indices <- sample(which(filtered), size=size_per_rank, replace=T)
    indices <- append(indices, sample_indices)
  }
  indices <- sample(indices) # Everyday I'm shuffling
  return(list(X = train_X[indices,],
              y = train_Y[indices]))
}


run_tree <- function(X,y, ...){
  # Runs random forest with fixed parameters (to ease the tuning accross all attempts)
  fit <- randomForest(X, y, ntree=150, nodesize = 5, ...)
  return(fit)
}


main.tree <- function(){
  # # Utility function to run the different methods of random forest models in order  
  # to compare them
  methods <- list(
    regression = 
      function(X, y, folds){
        print('RF - REGRESSION (plain old)')
        for (set in names(X)){
          X[[set]][is.na(X[[set]])] <- rowvalue.to.matrix(X[[set]], rowMeans(X[[set]], na.rm = T))[is.na(X[[set]])]
          #X[[set]][is.na(X[[set]])] <- colvalue.to.matrix(X[[set]], colMeans(X[[set]], na.rm = T))[is.na(X[[set]])]  
          #X[[set]][is.na(X[[set]])] <- 3  
        }       
        
        rf <- run_tree(X$train, y$train)
        
        preds <- as.numeric(predict(rf, X$valid))
        #hist(preds)
        #confusion.distribution(as.numeric(preds), y$valid)
        return(loss(y$valid, to_rank(preds)))
      },
    
    classification.resampled =
      function(X, y, folds){
        print('RF - CLASSIFICATION with equal proportion in train set + class weight')
        for (set in names(X)){
          X[[set]][is.na(X[[set]])] <- rowvalue.to.matrix(X[[set]], rowMeans(X[[set]], na.rm = T))[is.na(X[[set]])]
          #X[[set]][is.na(X[[set]])] <- colvalue.to.matrix(X[[set]], colMeans(X[[set]], na.rm = T))[is.na(X[[set]])]  
          #X[[set]][is.na(X[[set]])] <- 3  
        }
        
        rarity <- aggregate(rep(1, length(y$train)), by=list(y$train), FUN=sum) / length(y$train)
        w = rarity$x
        
        resampled.train <- resample(2000, X$train, y$train)
        
        rf <- run_tree(resampled.train$X, as.ordered(resampled.train$y), classwt = w)
        
        preds <- predict(rf, X$valid)
        #hist(as.numeric(preds))
        #confusion.distribution(as.numeric(preds), y$valid)
        #print(loss(y$valid, to_rank(as.numeric(preds), c(-Inf, -Inf, -Inf, 3, 4, Inf))))
        return(loss(y$valid, to_rank(as.numeric(preds))))
      },
    
    classification = 
      function(X, y, folds){
        print('RF - CLASSIFICATION for original train set + class weight')
        for (set in names(X)){
          X[[set]][is.na(X[[set]])] <- rowvalue.to.matrix(X[[set]], rowMeans(X[[set]], na.rm = T))[is.na(X[[set]])]
          #X[[set]][is.na(X[[set]])] <- colvalue.to.matrix(X[[set]], colMeans(X[[set]], na.rm = T))[is.na(X[[set]])]  
          #X[[set]][is.na(X[[set]])] <- 3  
        }
        
        rarity <- aggregate(rep(1, length(y$train)), by=list(y$train), FUN=sum) / length(y$train)
        w = rarity$x
        
        rf <- run_tree(X$train, as.ordered(y$train), classwt = w)
        
        preds <- predict(rf, X$valid)
        #hist(as.numeric(preds))
        #confusion.distribution(as.numeric(preds), y$valid)
        #print(loss(y$valid, to_rank(as.numeric(preds), c(-Inf, -Inf, -Inf, 3, 4, Inf))))
        return(loss(y$valid, to_rank(as.numeric(preds))))
      },
    
    regression.iterative =
      function(X, y, folds){
        print('RF - REGRESSION with imputer-forest for each movie')
        rarity <- aggregate(rep(1, length(y$train)), by=list(y$train), FUN=sum) / length(y$train)
        w = rarity$x
        
        impt.model = impute.iteratively(randomForest, X$train, y$train, as_ordered=T, 
                                        imput.args=list(ntree=10, nodesize=10),
                                        predict.args= list(classwt = w, ntree=100, nodesize=5)
        )
        return(loss(y$valid, impute.and.predict(impt.model, X$valid)))
      }
  )
  average.loss(10, methods)
}
