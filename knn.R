# KNN.R - collection of function to fit and predict KNN based modules
# Authors: Eran Amar & Tomer Levy


# ---- Distance functions ---- 

dist.l2 <- function(x1, x2){
  (norm(as.matrix(x1 - x2)))
}


gen.dist.weighted <- function(w) {
  # Generated distance according to given weights vector
  # see section 2.1.1 in the report.
  
  n <- length(w) + 1
  W <- matrix(0, nrow = n, ncol = n)
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      W[i, j] <-  W[i, j - 1] + w[j - 1]
    }
  }
  W <- W + t(W)
  dist.weighted <- function(x1, x2) {
    norm(as.matrix(apply(rbind(x1, x2), 2, function(coor)
      (W[coor[1], coor[2]]))))
  }
  return(dist.weighted)
}


# ---- KNN fit funcions  ---- 

knn_single_fit <-
  function(data_X, data_y, single_test_x, dist.fun) {
    # for given test vector (single_test_x) returns ordered data_y 
    # according to dist.fun on the corresponding rows in data_x
    
    active_set <- as.vector(!is.na(single_test_x))
    matching_users <- rowSums(is.na(data_X[, active_set])) == 0
    
    stopifnot(any(matching_users))
    
    neighbors <- data_X[matching_users, active_set]
    test_x <- single_test_x[active_set]
    distances <- apply(neighbors,
                       1,
                       function(row)
                         (dist.fun(row, test_x)))
    selected_neighbors <- order(distances)
    votes_indecies <- which(matching_users)[selected_neighbors]
    ordered_neighbor_votes <- data_y[votes_indecies]
    full_ordered_votes <-
      append(ordered_neighbor_votes, rep(NA, nrow(data_X) - nrow(neighbors)))
    return(full_ordered_votes)
  }

knn_multi_fit <- function(data_X, data_y, test_X, dist.fun) {
  # Extends knn_single_fit to a matrix (by rows)
  apply(test_X, 1, function(single_x)
    (knn_single_fit(data_X, data_y, single_x, dist.fun)))
}

knn.bagging.fit <-
  function(num_predictors,
           bootstrap_size,
           train_X,
           train_y,
           test_X,
           dist) {
    # fit num_predictors instances of KNN, each of different sampled subset of 
    # train_X where the number of rows sampled is bootstrap_size
    # returns list of all those predictors. Compatible with knn.bagging.predict()
    
    models <- list()
    for (i in 1:num_predictors) {
      ind <- sample(1:nrow(train_X), size = bootstrap_size, replace = T)
      models[[i]] <-
        knn_multi_fit(train_X[ind,], train_y[ind], test_X, dist)
    }
    return(models)
  }

# --- Predict function ---

knn.predict <- function(knn_fits, k) {
  if (k > 1) {
    return(colMeans(as.matrix(knn_fits[1:k, ]), na.rm = T))
  } else {
    return(knn_fits[1, ])
  }
}

knn.model_selection <-
  function(train_X,
           train_y,
           test_X,
           test_y,
           max_k,
           dist) {
    # fit KNN on train_X and evaluate predictions on test_X for best
    # k param in 1,..,max_k and return vector of losses for all the k's
    # Also plots the RMSE by k param
    
    knn.fits <- knn_multi_fit(train_x, train_y, test_X, dist)
    all_losses <- c()
    for (k in 1:max_k) {
      relevant_votes <- knn.predict(knn.fits, k)
      preds <- to_rank(relevant_votes)
      l <- loss(test_y, preds)
      print(paste('for k=', k, 'rmse is:', l))
      all_losses <- append(all_losses, l)
    }
    
    plot(1:max_k, all_losses, xlab = 'num of neighbors', ylab = 'RMSE on validation')
    title('RMSE by number of neighbors')
    best_k <- which.min(all_losses)
    print(
      paste(
        'best number of neighbors is k=',
        best_k,
        'with loss=',
        all_losses[best_k],
        '[Note that used NA.RM=T in knn.predict!]'
      )
    )
    return(all_losses)
  }

knn.bagging.predict <- function(model_list, k) {
  vote_per_predictor <-
    sapply(model_list, function(model)
      (knn.predict(model, k)))
  rowMeans(vote_per_predictor)
}

knn.bagging.model_selection <-
  function(num_predictors,
           bootstrap_size,
           train_X,
           train_y,
           test_X,
           test_y,
           dist) {
    # fit bagging KNN with num_predictors predictors and  bootstrap_size sample size 
    # on train_X and evaluate predictions on test_X for best k param in 1,..,max_k 
    # Returns vector of losses for all the k's. Also plots the RMSE by k param
    
    max_k <- bootstrap_size
    models <-
      knn.bagging.fit(num_predictors,
                      bootstrap_size,
                      train_X,
                      train_y,
                      test_X,
                      dist)
    
    all_losses = c()
    for (k in 1:max_k) {
      preds <- to_rank(knn.bagging.predict(models, k))
      l <- loss(test_y, preds)
      print(
        paste(
          'Bagging knn: for k=',
          k,
          ', bootstrap=',
          bootstrap_size,
          'num.predictors=',
          num_predictors,
          'rmse is:',
          l
        )
      )
      all_losses = append(all_losses, l)
    }
    
    plot(1:max_k, all_losses, xlab = 'num of neighbors', ylab = 'RMSE on validation')
    title('RMSE by number of neighbors [bagging settings]')
    best_k = which.min(all_losses)
    print(
      paste(
        'best number of neighbors is k=',
        best_k,
        'with loss=',
        all_losses[best_k],
        '[Note that used NA.RM=T!]'
      )
    )
    return(all_losses)
  }


# ---------- Main function ----------

main.knn <- function() {
  # Utility function to run the different methods of KKN in order to compare 
  # them
  average.loss(
    10,
    list(
      
      knn.na.ignore = function(X, y, folds) {
                        print("KNN fit ignoring NA")
                        model <- knn_multi_fit(X$train, y$train, X$valid, dist.l2)
                        preds <- knn.predict(model, 15)
                        return(loss(y$valid, to_rank(preds)))
                      },
      
      knn.na.ignore.centered.movie = function(X, y, folds) {
                                        print("KNN fit ignoring NA")
                                        X$train <- center_rating_per_movie(X$train)
                                        X$valid <- center_rating_per_movie(X$valid)
                                        model <- knn_multi_fit(X$train, y$train, X$valid, dist.l2)
                                        preds <- knn.predict(model, 15)
                                        return(loss(y$valid, to_rank(preds)))
                                      },
      
      knn.na.ignore.centered.user = function(X, y, folds) {
                                      print("KNN fit ignoring NA")
                                      X$train <- center_rating_per_person(X$train)
                                      X$valid <- center_rating_per_person(X$valid)
                                      model <- knn_multi_fit(X$train, y$train, X$valid, dist.l2)
                                      preds <- knn.predict(model, 15)
                                      return(loss(y$valid, to_rank(preds)))
                                    },
      
      knn.bagging.centered.user = function(X, y, folds) {
                                      print("KNN bagging model selection")
                                      X$train <- center_rating_per_person(X$train)
                                      X$test <- center_rating_per_person(X$test)
                                      X$valid <- center_rating_per_person(X$valid)
                                      losses <-
                                        knn.bagging.model_selection(15,
                                                                    300,
                                                                    fill.na(X$train, 3),
                                                                    y$train,
                                                                    fill.na(X$test, 3),
                                                                    y$test,
                                                                    dist.l2)
                                      k <- which.min(losses)
                                      
                                      print(paste("KNN by bagging with imputed by mode with selected k =", k))
                                      model <-
                                        knn.bagging.fit(15,
                                                        300,
                                                        fill.na(X$train, 3),
                                                        y$train,
                                                        fill.na(X$valid, 3),
                                                        dist.l2)
                                      preds <- knn.bagging.predict(model, k)
                                      return(loss(y$valid, to_rank(preds)))
                                    },
      
      knn.bagging.centered.movie = function(X, y, folds) {
                                      print("KNN bagging model selection")
                                      X$train <- center_rating_per_movie(X$train)
                                      X$test <- center_rating_per_movie(X$test)
                                      X$valid <- center_rating_per_movie(X$valid)
                                      losses <-
                                        knn.bagging.model_selection(15,
                                                                    300,
                                                                    fill.na(X$train, 3),
                                                                    y$train,
                                                                    fill.na(X$test, 3),
                                                                    y$test,
                                                                    dist.l2)
                                      k <- which.min(losses)
                                      
                                      print(paste("KNN by bagging with imputed by mode with selected k =", k))
                                      model <-
                                        knn.bagging.fit(15,
                                                        300,
                                                        fill.na(X$train, 3),
                                                        y$train,
                                                        fill.na(X$valid, 3),
                                                        dist.l2)
                                      preds <- knn.bagging.predict(model, k)
                                      return(loss(y$valid, to_rank(preds)))
                                    },
      
      knn.bagging.selection = function(X, y, folds) {
                                print("KNN bagging model selection")
                                losses <-
                                  knn.bagging.model_selection(15,
                                                              300,
                                                              fill.na(X$train, 3),
                                                              y$train,
                                                              fill.na(X$test, 3),
                                                              y$test,
                                                              dist.l2)
                                k <- which.min(losses)
                                
                                print(paste("KNN by bagging with imputed by mode with selected k =", k))
                                model <-
                                  knn.bagging.fit(15,
                                                  300,
                                                  fill.na(X$train, 3),
                                                  y$train,
                                                  fill.na(X$valid, 3),
                                                  dist.l2)
                                preds <- knn.bagging.predict(model, k)
                                return(loss(y$valid, to_rank(preds)))
                              }
      
    )
  )
}
