# final_model.R - Script for predicting the "Miss Congeniality" ratings per user based 
#                 on other movie ratings from the same user. Uses random forest to imput the 
#                 missing values and elastic net to predict the variable.
#
# Authors: Eran Amar & Tomer Levy

library(glmnet)
library(randomForest)
source('knn.R')
source('utils.R')
source('data_loader.R')

data <- load_data()

best.fit <- function(X, y, alpha = 1, ntree = 20, nodesize = 15){
  # The fit function
  glmnet.args <- list(alpha=alpha)
  rf.args <- list(ntree=ntree, nodesize = nodesize)

  imputed.model <- impute.iteratively(randomForest, X, y, as_ordered = F, 
                                      imput.args = rf.args, predict.args = glmnet.args, 
                                      predict.FUN=cv.glmnet)
  return(imputed.model)
}

best.predict <- function(imputed.model, X){
  # The predict function
  preds <- to_rank(impute.and.predict(imputed.model, X))
  return(preds)
}

model <- best.fit(as.matrix(data$train$X$rating), data$train$y$rating[,1])
preds <- best.predict(model, as.matrix(data$test$X$rating))

write.csv(preds, "Amar_Eran_Levy_Tomer_predictions.csv")
