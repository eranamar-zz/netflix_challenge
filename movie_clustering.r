# utils.R - collection of function to help cluster users and movies
#           by similarity.
# Authors: Eran Amar & Tomer Levy

source("utils.R")
source("data_loader.R")
library(glmnet)
library(gplots)

calc.cosine.sym <- function(X){
  # Calculate cosine similarity, accounting to rows with zero standard deviation.
  X <- as.matrix(X)
  X[is.na(X)] <- 0
  
  X.ss <-X %*% t(X)
  X.ss.sqrt <- sqrt(diag(X.ss))
  X.ss.sqrt[X.ss.sqrt == 0] <- 1
  X.cosine <- X.ss / (X.ss.sqrt%*% t(X.ss.sqrt))
  return(X.cosine)
}

calc.cosine.sym.centered <- function(train.ratings){
  # Centeromg tje data, then calculate cosine similarity.
  train.ratings.centered <- as.matrix(center_rating_per_person(train.ratings))
  return(calc.cosine.sym(t(train.ratings.centered)))
}


cluster.users <- function(train.ratings, k, center=T){
  # Cluster users into k clusters using cosine symmilarity.
  
  if (center){
    sym <- calc.cosine.sym(train.ratings-3.5)
  } else {
    sym <- calc.cosine.sym(train.ratings)
  }
  
  h <- hclust(as.dist(1/sym), method = "ward.D2")
  # Plotting the cluster
  plot(h)
  rect.hclust(h, k=k, border="red")
  # Splitting into groups
  groups <- cutree(h, k=k)
  return(groups)
}


by.group.train <- function(FUN, X, y, groups, ...){
  # Trains predictor to each group  in cluster seperately
  
  X.groups <- split.data.frame(X, groups)
  y.groups <- split(y, groups)
  models <- list()
  for (g in min(groups):max(groups)){
    models[[g]] <-FUN(X.groups[[g]], y.groups[[g]], ...)
  }
  return(models)
}

by.group.predict <- function(model, X, groups){
  # Predict by seperate predictor to each group in cluster
  
  X.groups <- split.data.frame(X, groups)
  preds <- list()
  for (g in min(groups):max(groups)){
    preds[[g]] <- predict(model[[g]], X.groups[[g]])
  }
  return(unsplit(preds, groups))
}



