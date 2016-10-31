library(dplyr)
library(slice)
library(xgboost)
library(ada)
library(plyr)
library(caret)
library(caretEnsemble)
library(randomForest)
library(rpart)
setwd("~/Documents/NYCDSA/Project 4")

### Subset DTest and calculate the csv file. the ranking must be done when everythinn is toghether
if (exists("dfTest") == F){
  # Check whether the Test data is loaded, otherwise load it and substitute -999
  dfTest = read.csv('test.csv')
  dfTest[dfTest==-999.0] = NA
  print('dfTest loaded!')
}

# impute missing columns for dfTest0
df0_TEST =  filter(dfTest,PRI_jet_num == 0)[, -c(6:8, 14, 25:30)]
df0_TEST$DER_mass_MMC = impute(df0_TEST$DER_mass_MMC, "random")

# impute missing columns for dfTest1
df1_TEST =  filter(dfTest,PRI_jet_num == 1)[, -c(6:8, 14, 28:30)]
df1_TEST$DER_mass_MMC = impute(df1_TEST$DER_mass_MMC, "random")

# impute missing columns for dfTest23
df23_TEST = dfTest %>% filter(.,(PRI_jet_num == 2) | (PRI_jet_num == 3) )
df23_TEST$DER_mass_MMC = impute(df23_TEST$DER_mass_MMC, "random")


########################################################
# Load
########################################################

# Load help functions
source('helper.r')

#df0
load("data/model_list_big.Rdata")
load("data/logit.overall0.Rdata")
#df1:  
load("data/df1_model_lc.Rdata")
load('data/logit.overall1.Rdata')
#df2:
load("data/df23_model.RData")
load('data/logit.overall23.Rdata')

#######################################################
traindf0 <- select(df0_TEST, c(-EventId, -PRI_jet_num, -PRI_jet_all_pt))
model0_preds <- lapply(model_list_big, predict, newdata=traindf0, type="prob")
design0<-cbind(model0_preds$ada[,2], model0_preds$ada.1[,2], 
               model0_preds$xgbtree[,2], model0_preds$xgbTree[,2],
               model0_preds$rf[,2], model0_preds$rf.1[,2])


predict0<- predict(logit.overall0, as.data.frame(design0), type = "response" )
hist(predict0)

# Apply threshld for df0

class0 <- ifelse(predict0> 0.988, 1, 0)

save(class0, file='data/df0_TESTens_LC.Rdata')

##########################################################
# DF1
##########################################################

traindf1 <- select(df1_TEST, c(-EventId, -PRI_jet_num))
model1_preds <- lapply(model_list_df1, predict, newdata=traindf1, type="prob")
design1<-cbind(model1_preds$ada[,2], model1_preds$ada.1[,2], 
               model1_preds$xgbtree[,2], model1_preds$xgbTree[,2],
               model1_preds$rf[,2], model1_preds$rf.1[,2])


predict1<- predict(logit.overall1, as.data.frame(design1), type = "response" )
hist(predict1)

# Apply threshld for df1

class1 <- ifelse(predict1> 0.64, 1, 0)

save(class1, file='data/df1_TESTens_LC.Rdata')

#2

##########################################################
# DF23
##########################################################

traindf23 <- select(df23_TEST, c(-EventId, -PRI_jet_num))
model23_preds <- lapply(df23_model, predict, newdata=traindf23, type="prob")
design23<-cbind(model23_preds$ada[,2], model23_preds$ada.1[,2], 
               model23_preds$xgbtree[,2], model23_preds$xgbTree[,2],
               model23_preds$rf[,2], model23_preds$rf.1[,2])


predict23<- predict(logit.overall23, as.data.frame(design23), type = "response" )
hist(predict23)

# Apply threshld for df1

class23 <- ifelse(predict23> 0.988, 1, 0)

save(class23, file='data/df23_TESTens_LC.Rdata')

################################################################################
################################################################################
#final merge of df0, 1, 23, prediction on the test.csv

# merging classes
df0123_ens_solution = data.frame('EventId' = dfTest$EventId, 'RankOrder'= seq(550000),
                                 'Class' = c(class0,class1,class23), 
                                 'Raw_Probs' = c(predict0,predict1,predict23))

# Change class into categorical

df0123_ens_solution$Class = ifelse(df0123_ens_solution$Class == 0,'b','s')
df0123_ens_solution = mutate(df0123_ens_solution,RankOrder = as.integer(rank(Raw_Probs,ties.method = 'random'))) %>%
  select(.,-Raw_Probs)

# # Merging all solutions
# df0123_ens_solution = bind_rows(df0_to_be_merged, df1_ens_solution, df23_ens_solution) %>%
#   mutate(.,RankOrder = as.integer(rank(Raw_Probs,ties.method = 'random'))) %>%
#   select(.,-Raw_Probs)

# save(df0123_ens_solution,file = 'dfTest_predictions.Rdata')

# save it to csv
write.csv(df0123_ens_solution, file = 'data/3buddiessolution.csv',row.names = FALSE)
