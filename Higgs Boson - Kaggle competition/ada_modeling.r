devtools::install_github("topepo/caret/pkg/caret")
library(caret)
library(xgboost)
library(Ckmeans.1d.dp)
library(dplyr)
library(tidyr)
library(ggplot2)
library(doMC)
registerDoMC(cores = 2)
source('helper.r')

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

formula = paste(names(train)[2:31],sep = "  ", collapse = ' + ')
formula = paste('Label ~ ',formula)

ada.model <- ada(y = factor(train$Label), x = train[,2:31], 
                 test.y = factor(test$Label), test.x = test[,2:31], type = "real",
                    control = rpart.control(maxdepth=c(10),cp=-1,minsplit=0,xval=0),
                    iter = 100, nu = .03, bag.frac = 0.1)

save(ada.model,file = "ADA_model.RData")
plot(ada.model)

threshold <- print.prediction(ada.model ,test, type='ada')

if (exists("dfTest") == F){
  # Check whether the Test data is loaded, otherwise load it and substitute -999
  dfTest = read.csv('test.csv')
  dfTest[dfTest==-999.0] = NA
  print('dfTest loaded!')
}

output_ada = make_solution(ada.model, threshold, type = 'ada','tuned')


