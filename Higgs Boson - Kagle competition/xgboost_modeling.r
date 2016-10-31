# devtools::install_github("topepo/caret/pkg/caret")
library(caret)
library(xgboost)
library(Ckmeans.1d.dp)
library(dplyr)
library(tidyr)
library(ggplot2)
library(doMC)
registerDoMC(cores = 2)

########################################################
# Load
########################################################
setwd("~/Documents/NYCDSA/Project 4")

dfTrain = read.csv('training.csv')
summary(dfTrain)
head(dfTrain)

# Setting 1 for 's', 0 for 'b'
dfTrain$Label = ifelse(dfTrain$Label == 'b',0,1)

# Preparing train/test data
dfTrain[dfTrain==-999.0] = NA
#use create Data Partion in R caret for stratified sampling
train.index = createDataPartition(dfTrain$Label, p = .8, list = FALSE)

ind.row = train.index
ind.col = c()
train <- dfTrain[ind.row,]
test  <- dfTrain[-ind.row,]

# Create xbg matrices
dtrain <- xgb.DMatrix(data = as.matrix(train[,2:31]), label = train$Label,missing = NA)
dtest = xgb.DMatrix(data = as.matrix(test[,2:31]), label = test$Label,missing = NA)

####################
# Funs:
####################

print.prediction = function(model,test){
  
  # pred <- predict(model, as.matrix(test[,2:31]),missing=NA)
  pred.threshold = AMS.cv(model,test)
  # Apply threshold
  pred = ifelse(pred.threshold$pred < pred.threshold$best, 0,1)
  # calculate tab
  tab = table(round(pred), test$Label)
  print(tab)
  print(paste0('Length = ',length(pred),', accuracy = ', (tab[1,1]+tab[2,2])/sum(tab)))
  return(pred)
}

plot.importance = function(input){
  importance_matrix <- xgb.importance(model = input)

  print(importance_matrix)
  xgb.plot.importance(importance_matrix = importance_matrix)
  }

make_solution = function(model, threshold, comment = 'test'){
  
  # Calculate final predictions
  pred = predict(model, as.matrix(dfTest[,2:31]),missing=NA)
    
  # Calculate final ranking
  rank.pred = as.integer(rank(pred,ties.method = 'random'))
  if (length(unique(rank.pred)) != length(pred)){
    return(print('Wrong ranking'))}

  # create submission file
  df.solution = select(dfTest, EventId) %>% bind_cols(.,
                  data.frame('RankOrder' = rank.pred, 'Class' = pred)) %>%
    mutate(.,Class = ifelse(Class < threshold,'b','s'))

  # save it to csv
  write.csv(df.solution, paste0('xgboost_',comment,'.csv'),row.names = FALSE)

  return(df.solution)
}
####### The AMS function defined according to the evaluation page on the website
AMS = function(real,pred,weight){
  pred_s_ind = which(pred==1)                          # Index of s in prediction
  real_s_ind = which(real==1)                          # Index of s in actual
  real_b_ind = which(real==0)                          # Index of b in actual
  s = sum(weight[intersect(pred_s_ind,real_s_ind)])      # True positive rate
  b = sum(weight[intersect(pred_s_ind,real_b_ind)])      # False positive rate

  b_tau = 10                                             # Regulator weight
  ans = sqrt(2*((s+b+b_tau)*log(1+s/(b+b_tau))-s))
  return(ans)
}

####### Using AMS function defined above as a metrics function for caret
####### Check the details here: http://topepo.github.io/caret/training.html#metrics
AMS_summary <- function(data, lev = NULL, model = NULL){
  out = (AMS(data$obs, data$pred, data$weights))
  names(out) <- "AMS"
  return(out)
}

AMS.cv = function(model, test){
  # threshold range
  threshold = seq(0.5,1,0.05)
  score = threshold
  # Calculate predictions
  pred = predict(model, as.matrix(select(test,c(-EventId,-Weight,-Label))),missing=NA)
  
  for(item in seq(1,length(threshold))){
    tmp = ifelse(pred < threshold[item],0,1)
    score[item] = AMS(test$Label,tmp,test$Weight)
  }
  # print(score)
  best = threshold[score == max(score)]
  plot(x = threshold, y = score)
  print(paste0('Best threshold is ',best))
  
  return(list("pred" = pred, 'best'=best))
}
#######################################################
# TRAINING simple
#######################################################

bstDMatrix <- xgboost(data = dtrain, max.depth = 10, eta = .01, nthread = 2, nround = 100, 
                      objective = "binary:logistic", verbose = 2)

pred = print.prediction(bstDMatrix,test)
plot.importance(bstDMatrix)

# Just for fun. Try predict the test and create submission file
output = make_solution(bstDMatrix,0.85)

#######################################################
# Training and testing
#######################################################
dtrainW <- xgb.DMatrix(data = as.matrix(train[,2:31]), label = train$Label,
                     missing = NA)
param <- list(eta = .03,
              max_depth = 10,
              silent = 1,
              nthread = 16,
              subsample = .95,
              gamma = .1,
              verbose = 2,
              objective = "binary:logistic")

watchlist <- list(train = dtrainW, test=dtest)

xgb.model <- xgb.train(params = param, data=dtrainW, nround=1000, stratified = TRUE,
                         watchlist=watchlist)

save(xgb.model,file = "xgb_model.RData")

threshold <- print.prediction(xgb.model,test)

plot.importance(xgb.model)

if (exists("dfTest") == F){
  # Check whether the Test data is loaded, otherwise load it and substitute -999
  dfTest = read.csv('test.csv')
  dfTest[dfTest==-999.0] = NA
  print('dfTest loaded!')
  }

output = make_solution(xgb.model, threshold)

#######################################################
# Training after tuning
#######################################################

load("xgb_cv.RData")
tune = xgb.tune$bestTune
param = list(eta = tune$eta,
             max_depth = tune$max_depth,
             nthread = 16,
             subsample = .95,
             gamma = tune$gamma,
             verbose = 2,
             colsample_bytree = tune$colsample_bytree,
             min_child_weight = tune$min_child_weight,
             objective = "binary:logistic")

watchlist <- list(train = dtrain, test=dtest)

bstDMatrix.tune <- xgb.train(params = param, data=dtrain, nround=tune$nrounds, stratified = TRUE,
                        watchlist=watchlist)


threshold <- print.prediction(bstDMatrix.tune,test)

plot.importance(bstDMatrix.tune)

if (exists("dfTest") == F){
  # Check whether the Test data is loaded, otherwise load it and substitute -999
  dfTest = read.csv('test.csv')
  dfTest[dfTest==-999.0] = NA
  print('dfTest loaded!')
}

xgb_output = make_solution(bstDMatrix.tune, threshold,'tuned')



