#manual caculation for the *** stakc:

library(dplyr)
library(xgboost)
library(ada)
library(plyr)
library(caret)
library(caretEnsemble)
library(randomForest)
library(rpart)

setwd("~/Documents/NYCDSA/Project 4")
########################################################
# Load
########################################################
# Load  subsets and clean them
source('Data_cleaning.r')

# Load help functions
source('helper.r')

#0
load("data/model_list_big.Rdata")

traindf0 <- select(df0_im_train, c(-EventId, -PRI_jet_num, -PRI_jet_all_pt,-Weight, -Label))
    testdf0<- select(df0_im_test, c(-EventId, -PRI_jet_num, -PRI_jet_all_pt,-Weight, -Label))

model_preds0 <- lapply(model_list_big, predict, newdata=traindf0, type="prob")
    model_tune0 <- lapply(model_list_big, predict, newdata=testdf0, type="prob")

design0<-cbind(model_preds0$ada[,2], model_preds0$ada.1[,2], 
               model_preds0$xgbtree[,2], model_preds0$xgbTree[,2],
               model_preds0$rf[,2], model_preds0$rf.1[,2])

     design0.tune<-cbind(model_tune0$ada[,2], model_tune0$ada.1[,2], 
                         model_tune0$xgbtree[,2], model_tune0$xgbTree[,2],
                         model_tune0$rf[,2], model_tune0$rf.1[,2])

logit.data0<-as.data.frame(cbind(design0, df0_im_train$Label))
     logit.data0.tune <-as.data.frame(cbind(design0.tune, df0_im_test$Label))

#use logit to pull together:
logit.overall0 = glm(V7 ~ . ,
                    family = "binomial",
                    data = logit.data0)

save(logit.overall0, file='data/logit.overall0.Rdata')

# tune, there is no time to tune against test, but...
   predict.tune<- predict(logit.overall0, logit.data0.tune, type = "response" )
   hist(predict.tune)
   

#search for the best threshold against AWS, 
   
accuracy.ams=c()

for (i in seq(0.9, 1, by = 0.001)){
  pred_tmp<-ifelse(predict.tune>i, 1, 0)
  accuracy.ams<-AMS(df0_im_test$Label, pred_tmp, df0_im_test$Weight)
  print(c(accuracy.ams,i)) #i refers to the cutoff 
}

 #0.988...

class <- ifelse(predict.tune> 0.988, 1, 0)
table(class, df0_im_test$Label)  

# class     0     1
# 0 14869  1614
# 1     6  3493

#Accuracy: [1] 0.919203

save(class, file='data/df0_ens_LC.Rdata')

#########################################
# DF1
#########################################
load("data/df1_model_lc.Rdata")

traindf1 <- select(df1_im_train, c(-EventId, -PRI_jet_num, -Label, -Weight))
    testdf1<- select(df1_im_test, c(-EventId, -PRI_jet_num, -Weight, -Label))

model_preds1 <- lapply(model_list_df1, predict, newdata=traindf1, type="prob")
    model_tune1 <- lapply(model_list_df1, predict, newdata=testdf1, type="prob")

design1<-cbind(model_preds1$ada[,2], model_preds1$ada.1[,2], 
               model_preds1$xgbtree[,2], model_preds1$xgbTree[,2],
               model_preds1$rf[,2], model_preds1$rf.1[,2])

design1.tune<-cbind(model_tune1$ada[,2], model_tune1$ada.1[,2], 
                    model_tune1$xgbtree[,2], model_tune1$xgbTree[,2],
                    model_tune1$rf[,2], model_tune1$rf.1[,2])

logit.data1<-as.data.frame(cbind(design1, df1_im_train$Label))
logit.data1.tune <-as.data.frame(cbind(design1.tune, df1_im_test$Label))

#use logit to pull together:
logit.overall1 = glm(V7 ~ . ,
                    family = "binomial",
                    data = logit.data1)
save(logit.overall1, file='data/logit.overall1.Rdata')

# tune, there is no time to tune against test, but...
predict.tune1<- predict(logit.overall1, logit.data1.tune, type = "response" )
hist(predict.tune1)


#search for the best threshold against AWS, 

accuracy.ams=c()

for (i in seq(0.01, 1, by = 0.01)){
  pred_tmp<-ifelse(predict.tune1>i, 1, 0)
  accuracy.ams<-AMS(df1_im_test$Label, pred_tmp, df1_im_test$Weight)
  print(c(accuracy.ams,i)) #i refers to the cutoff 
}

#0.64...

class1 <- ifelse(predict.tune1> 0.64, 1, 0)
table(class1, df1_im_test$Label)  

#class1    0    1
#0 8477 2407
#1 1442 3182
#   [1] 0.7518055    #df1 is always the worst...
save(class1, file='data/df1_ens_LC.Rdata')
#####################################################
# DF23
#####################################################
load("data/df23_model.RData")

traindf23 <- select(df23_im_train, c(-EventId, -PRI_jet_num, -Weight, -Label))
testdf23<- select(df23_im_test, c(-EventId, -PRI_jet_num, -Weight, -Label))

model_preds23 <- lapply(df23_model, predict, newdata=traindf23, type="prob")
model_tune23 <- lapply(df23_model, predict, newdata=testdf23, type="prob")

design23<-cbind(model_preds23$ada[,2], model_preds23$ada.1[,2], 
               model_preds23$xgbtree[,2], model_preds23$xgbTree[,2],
               model_preds23$rf[,2], model_preds23$rf.1[,2])

design23.tune<-cbind(model_tune23$ada[,2], model_tune23$ada.1[,2], 
                    model_tune23$xgbtree[,2], model_tune23$xgbTree[,2],
                    model_tune23$rf[,2], model_tune23$rf.1[,2])

logit.data23<-as.data.frame(cbind(design23, df23_im_train$Label))
logit.data23.tune <-as.data.frame(cbind(design23.tune, df23_im_test$Label))

#use logit to pull together:
logit.overall23 = glm(V7 ~ . ,
                    family = "binomial",
                    data = logit.data23)

save(logit.overall23, file='data/logit.overall23.Rdata')

# tune, there is no time to tune against test, but...
predict.tune23<- predict(logit.overall23, logit.data23.tune, type = "response" )
hist(predict.tune23, breaks = 100)

#search for the best threshold against AWS, 

accuracy.ams=c()

for (i in seq(0.99, 1, by = 0.0001)){
  pred_tmp<-ifelse(predict.tune23>i, 1, 0)
  accuracy.ams <- AMS(df23_im_test$Label, pred_tmp, df23_im_test$Weight)
  print(c(accuracy.ams,i)) #i refers to the cutoff 
}

#0.99...

class2 <- ifelse(predict.tune23> 0.99, 1, 0)
tab2 = table(class2, df23_im_test$Label)  
(tab2[1,1]+tab2[2,2])/length(class2)

save(class2, file='data/df23_ens_LC.Rdata')

# class     0     1
# 0 14869  1614
# 1     6  3493

#Accuracy: [1] 0.919203