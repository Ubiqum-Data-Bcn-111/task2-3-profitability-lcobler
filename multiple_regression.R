#Task 2.3: Multiple regression in R
#Lara Cobler Moncunill
#November 15th, 2018

library(readr)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(plotly)
library(plot3D)

#download data set
existing <- read.csv("existingproductattributes2017.2.csv",header=TRUE)
str(existing)
summary(existing)

#Missing values for Ranking -> remove from the analysis
existing$BestSellersRank <- NULL

#check outliers other variables and normality
for (i in 1:(ncol(existing))){
  if (is.numeric(existing[,i])){
    boxplot(existing[,i],main=paste("Boxplot of",colnames(existing)[i]),ylab=colnames(existing)[i])
    qqnorm(existing[,i],main=paste("Normal Q-Q Plot of",colnames(existing)[i])) #plot qqnorm
    qqline(existing[,i],col="red") #add qqnormal line in red
    hist(existing[,i],main=paste("Histogram of",colnames(existing)[i]), #make the histogram
         xlab=colnames(existing)[i])
  }
}

#2 outliers volume > 6000
#remove outliers of volume
existing_out <- existing[existing$Volume<6000,]
boxplot(existing_out$Volume,main="Boxplot of volume",ylab="Volume")
boxplot(existing$Volume,main="Boxplot of volume",ylab="Volume")

#Product number: no sense in  the analysis
existing_out$ProductNum <- NULL

#remove waranty that are the same volume and same reviews/duplicates
existing_out <-existing_out %>% distinct(Volume, x5StarReviews,x4StarReviews,NegativeServiceReview, #removes the duplicates from variables specified
                                         .keep_all = TRUE) #keeps the variables
#OR
existing_out2 <- existing[!duplicated(existing[,4:17]),]

#Build the correlation matrix
corr_existing_out <- cor(existing_out[,2:16]) #no include product type
corr_existing_out
#nice plot of the correlation matrix
corrplot(corr_existing_out,order="hclust",type="upper")

#filter the correlation matrix
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}
pmat<- cor.mtest(existing_out[,2:16])
corrplot(corr_existing_out,order="hclust",p.mat=pmat,sig.level=.05)

#remove variables that correlate between them
#5star,3star, 2star, 1star, negative service review, width and height
existing_out$x5StarReviews <- NULL
existing_out_less <- subset(existing_out,select=-c(x3StarReviews,x2StarReviews,x1StarReviews,
                                                  NegativeServiceReview,ProductWidth
                                                   ,ProductHeight))
#Anova to see if the product type is related to volume
anova_res <- aov(Volume~ProductType,data=existing_out)
summary(anova_res)
#No.

#variable importance of the left
set.seed(333)
fit_rf <- randomForest(Volume~.,data=existing_out_less)
(VI_F=importance(fit_rf))
varImpPlot(fit_rf,type=2)
#oreder: positive service, 4x, product type,  product depth, price, shipping weight,recommend,profit margin

#dummify the data, convert all factor or 'chr' classes to binary features
existing_dummy <- dummyVars("~.",data=existing_out_less)
existing_dum <- data.frame(predict(existing_dummy,newdata = existing_out_less))
str(existing_dum)

#variable importance with dummies
set.seed(333)
fit_rf_d <- randomForest(Volume~.,data=existing_dum)
(VI_F=importance(fit_rf_d))
varImpPlot(fit_rf_d,type=2)
#First the reviews, last product types separate, finally smartphones,PC,laptop, netbook, no important

#do one dummy column wit PC, laptops, notebooks, and smartphones
existing_dum <- transform(existing_dum,ProductType.PLNS=ProductType.Laptop+ProductType.Netbook+ProductType.PC+ProductType.Smartphone)
#remove other dummies of the data frame
existing_dum_PLNS <- existing_dum[,13:21]


palette<-c("red","black")
#Check relationship other variables vs volume and product type
for (i in 1:(ncol(existing_dum_PLNS))){
  plot (existing_dum_PLNS[,i],existing_dum_PLNS$Volume,xlab=colnames(existing_dum_PLNS)[i],ylab="Volume",
        col=palette[as.factor(existing_dum_PLNS$ProductType.PLNS)])
  abline (lm(Volume ~ existing_dum_PLNS[,i] , data=existing_dum_PLNS),col="blue") #add regression line
  legend(x="top", legend = c("Other","PLNS"), col=palette, pch=1)
}

#Variables order to add to modelling: 
#positive service, 4 star,product type, shipping weight, product depth, price and would recommend

#Feature selection using rfe in caret
control_rfe <- rfeControl(functions = rfFuncs,
                          method = "repeatedcv",
                          repeats = 3,
                          verbose = FALSE)
outcomeName<-'Volume'
predictors<-names(existing_dum_PLNS)[!names(existing_dum_PLNS) %in% outcomeName]
Volume_Pred_Profile <- rfe(existing_dum_PLNS[,predictors], existing_dum_PLNS[,outcomeName],
                         rfeControl = control_rfe)
Volume_Pred_Profile


# top5 existing_dum_PLNS: PositiveServiceReview, x4StarReviews,  ShippingWeight, Price, ProductDepth

# define an 75%/25% train/test split of the dataset
set.seed(123)
inTraining <- createDataPartition(existing_dum_PLNS$Volume, p = .75, list = FALSE)
#creates a vector with the rows to use for training
training <- existing_dum_PLNS[inTraining,] #subset training set
testing <- existing_dum_PLNS[-inTraining,] #subset testing set

#Check if the training and testing are equally distributed by volume
summary(training$Volume)
summary(testing$Volume)

par(mfrow=c(1,2))
boxplot(training$Volume, main="Training")
boxplot(testing$Volume, main="Testing")

#linear model
#linear model Positive service review and 4starreviews
LinearModel<- lm(Volume ~ PositiveServiceReview+x4StarReviews, training)
summary(LinearModel)
  # R-squared: 0.68, p-value: 6.677e-13
plot(LinearModel)
#prediction test set
predictions_LinearModel <- predict(LinearModel,testing)
postResample(predictions_LinearModel, testing$Volume)
#RMSE    Rsquared         MAE 
#332.8147633   0.7030515 201.4916643
plot(testing$Volume,predictions_LinearModel)
abline (a=0,b=1,col=2)

realErrolm<- (predictions_LinearModel -testing$Volume)/testing$Volume  # to calc %
realErrolm
mean(realErrolm) #relative error = 0.3277584


#xgbm in caret
modelLookup("xgbTree") #description parameters to tune

#start checking 1000 trees
nrounds <- 1000
tune_grid <- expand.grid(
  nrounds = seq(from=200, to = nrounds, by=50), #from 200 trees to 1000 50 by 50
  eta= c(0.025,0.05,0.1,0.3), #learning rate
  max_depth=c(2,3,4,5,6),
  gamma=0,
  colsample_bytree=1,
  min_child_weight=1,
  subsample=1
)

control_train <- trainControl(method = "repeatedcv", number = 10, repeats = 3) 
set.seed(123)
xgb_tune <- train(Volume~x4StarReviews+PositiveServiceReview, data=training,
                  method="xgbTree",
                  trControl=control_train,
                  tuneGrid=tune_grid)
plot(xgb_tune)
xgb_tune$bestTune
# best:nrounds = 250, max_depth = 2, eta = 0.1, gamma =0, 
# colsample_bytree = 1, min_child_weight = 1 and subsample = 1

#fix best learning rate:0.1 
#find max depth and minimum child weight
tune_grid2 <- expand.grid(
  nrounds = seq(from=50, to = nrounds, by=50), #from 50 trees to 1000 50 by 50
  eta= xgb_tune$bestTune$eta, #best learning rate previous modelling
  max_depth=(xgb_tune$bestTune$max_depth - 1:xgb_tune$bestTune$max_depth + 1), 
  gamma=0,
  colsample_bytree=1,
  min_child_weight=c(1,2,3),
  subsample=1
)
set.seed(123)
xgb_tune2 <- train(Volume~x4StarReviews+PositiveServiceReview, data=training,
                  method="xgbTree",
                  trControl=control_train,
                  tuneGrid=tune_grid2)
plot(xgb_tune2)
xgb_tune2$bestTune
#best: nrounds = 600, max_depth = 1, eta = 0.1, gamma =0, 
#colsample_bytree = 1, min_child_weight = 1 and subsample = 1.

#fix the child weigth 1 and max depth 1.
# find row and column sampling
tune_grid3 <- expand.grid(
  nrounds = seq(from=50, to = nrounds, by=50), #from 50 trees to 1000 50 by 50
  eta= xgb_tune$bestTune$eta, #best learning rate previous modelling
  max_depth=xgb_tune2$bestTune$max_depth, 
  gamma=0,
  colsample_bytree=c(0.4,0.6,0.8,1.0),
  min_child_weight=xgb_tune2$bestTune$min_child_weight,
  subsample=c(0.5,0.75,1.0)
)
set.seed(123)
xgb_tune3 <- train(Volume~x4StarReviews+PositiveServiceReview, data=training,
                   method="xgbTree",
                   trControl=control_train,
                   tuneGrid=tune_grid3)
plot(xgb_tune3)
xgb_tune3$bestTune
#row =0.75 and column =1
#fix the gamma
tune_grid4 <- expand.grid(
  nrounds = seq(from=50, to = nrounds, by=50), #from 50 trees to 1000 50 by 50
  eta= xgb_tune$bestTune$eta, #best learning rate previous modelling
  max_depth=xgb_tune2$bestTune$max_depth, 
  gamma=c(0,0.05,0.1,0.5,0.7,0.9,1.0),
  colsample_bytree=xgb_tune3$bestTune$colsample_bytree,
  min_child_weight=xgb_tune2$bestTune$min_child_weight,
  subsample=xgb_tune3$bestTune$subsample
)
set.seed(123)
xgb_tune4 <- train(Volume~x4StarReviews+PositiveServiceReview, data=training,
                   method="xgbTree",
                   trControl=control_train,
                   tuneGrid=tune_grid4)
plot(xgb_tune4)
xgb_tune4$bestTune
#best gamma=0.05
# reduce learning rate
tune_grid5 <- expand.grid(
  nrounds = seq(from=100, to = 10000, by=100), #from 100 trees to 10000 by 100
  eta= c(0.01,0.015,0.025,0.05,0.1), #best learning rate previous modelling
  max_depth=xgb_tune2$bestTune$max_depth, 
  gamma=xgb_tune4$bestTune$gamma,
  colsample_bytree=xgb_tune3$bestTune$colsample_bytree,
  min_child_weight=xgb_tune2$bestTune$min_child_weight,
  subsample=xgb_tune3$bestTune$subsample
)
set.seed(123)
xgb_tune5 <- train(Volume~x4StarReviews+PositiveServiceReview, data=training,
                   method="xgbTree",
                   trControl=control_train,
                   tuneGrid=tune_grid5)
plot(xgb_tune5)
xgb_tune5$bestTune
#nrounds = 300, max_depth = 1, eta = 0.1, gamma =0.05, 
#colsample_bytree = 1, min_child_weight = 1 and subsample = 0.75.
# RMSE:220.0111 Rsquared:0.9188357 MAE:131.3699 

##make predictions
test_xgb_tune5 <- predict(xgb_tune5, newdata=testing)
#performace measurment
postResample(test_xgb_tune5, testing$Volume)
#RMSE=159.047728, Rsquared= 0.968368, MAE=91.819854 

#Relative error
realErro<- (test_xgb_tune5 -testing$Volume)/testing$Volume  # to calc %
realErro
mean(realErro) #relative error = 0.3277584

#plot predicted verses real
palette=c("Black","Green")
plot(test_xgb_tune5,testing$Volume, col=palette[as.factor(testing$ProductType.PLNS)])  #good for lower values (volume <500)
abline(a=0,b=1,col="red")
legend(x="bottomright", legend = c("Other","PLNS"), col=palette, pch=1)
#Good model for the products we want to predict! 


#add shipping wheight to prediction
#start checking 1000 trees
nrounds <- 1000
tune_grid_w <- expand.grid(
  nrounds = seq(from=200, to = nrounds, by=50), #from 200 trees to 1000 50 by 50
  eta= c(0.025,0.05,0.1,0.3), #learning rate
  max_depth=c(2,3,4,5,6),
  gamma=0,
  colsample_bytree=1,
  min_child_weight=1,
  subsample=1
)

control_train <- trainControl(method = "repeatedcv", number = 10, repeats = 3) 
set.seed(123)
xgb_tune_w <- train(Volume~x4StarReviews+PositiveServiceReview+ShippingWeight, data=training,
                  method="xgbTree",
                  trControl=control_train,
                  tuneGrid=tune_grid_w)
plot(xgb_tune_w)
xgb_tune_w$bestTune
# best:nrounds = 250, max_depth = 2, eta = 0.025, gamma =0, 
# colsample_bytree = 1, min_child_weight = 1 and subsample = 1

#find max depth and minimum child weight
tune_grid2_w <- expand.grid(
  nrounds = seq(from=50, to = nrounds, by=50), #from 50 trees to 1000 50 by 50
  eta= xgb_tune_w$bestTune$eta, #best learning rate previous modelling
  max_depth=(xgb_tune_w$bestTune$max_depth - 1:xgb_tune_w$bestTune$max_depth + 1), 
  gamma=0,
  colsample_bytree=1,
  min_child_weight=c(1,2,3),
  subsample=1
)
set.seed(123)
xgb_tune2_w <- train(Volume~x4StarReviews+PositiveServiceReview+ShippingWeight, data=training,
                   method="xgbTree",
                   trControl=control_train,
                   tuneGrid=tune_grid2_w)
plot(xgb_tune2_w)
xgb_tune2_w$bestTune
#best: nrounds = 1000, max_depth = 1, eta = 0.025, gamma =0, 
#colsample_bytree = 1, min_child_weight = 1 and subsample = 1.

#fix the child weigth 1 and max depth 1.
# find row and column sampling
tune_grid3_w <- expand.grid(
  nrounds = seq(from=50, to = nrounds, by=50), #from 50 trees to 1000 50 by 50
  eta= xgb_tune_w$bestTune$eta, #best learning rate previous modelling
  max_depth=xgb_tune2_w$bestTune$max_depth, 
  gamma=0,
  colsample_bytree=c(0.4,0.6,0.8,1.0),
  min_child_weight=xgb_tune2_w$bestTune$min_child_weight,
  subsample=c(0.5,0.75,1.0)
)
set.seed(123)
xgb_tune3_w <- train(Volume~x4StarReviews+PositiveServiceReview+ShippingWeight, data=training,
                   method="xgbTree",
                   trControl=control_train,
                   tuneGrid=tune_grid3_w)
plot(xgb_tune3_w)
xgb_tune3_w$bestTune
#row =1 and column =.8
#fix the gamma
tune_grid4_w <- expand.grid(
  nrounds = seq(from=50, to = nrounds, by=50), #from 50 trees to 1000 50 by 50
  eta= xgb_tune_w$bestTune$eta, #best learning rate previous modelling
  max_depth=xgb_tune2_w$bestTune$max_depth, 
  gamma=c(0,0.05,0.1,0.5,0.7,0.9,1.0),
  colsample_bytree=xgb_tune3_w$bestTune$colsample_bytree,
  min_child_weight=xgb_tune2_w$bestTune$min_child_weight,
  subsample=xgb_tune3_w$bestTune$subsample
)
set.seed(123)
xgb_tune4_w <- train(Volume~x4StarReviews+PositiveServiceReview+ShippingWeight, data=training,
                   method="xgbTree",
                   trControl=control_train,
                   tuneGrid=tune_grid4_w)
plot(xgb_tune4_w)
xgb_tune4_w$bestTune
#best gamma=0
# reduce learning rate
tune_grid5_w <- expand.grid(
  nrounds = seq(from=100, to = 10000, by=100), #from 100 trees to 10000 by 100
  eta= c(0.01,0.015,0.025,0.05,0.1), #best learning rate previous modelling
  max_depth=xgb_tune2_w$bestTune$max_depth, 
  gamma=xgb_tune4_w$bestTune$gamma,
  colsample_bytree=xgb_tune3_w$bestTune$colsample_bytree,
  min_child_weight=xgb_tune2_w$bestTune$min_child_weight,
  subsample=xgb_tune3_w$bestTune$subsample
)
set.seed(123)
xgb_tune5_w <- train(Volume~x4StarReviews+PositiveServiceReview+ShippingWeight, data=training,
                   method="xgbTree",
                   trControl=control_train,
                   tuneGrid=tune_grid5_w)
plot(xgb_tune5_w)
xgb_tune5_w$bestTune
#nrounds = 1100, max_depth = 1, eta = 0.05, gamma =0, 
#colsample_bytree = 0.8, min_child_weight = 1 and subsample = 1.
#RMSE=216.7161, Rsquared= 0.9249764 MAE:130.5042 

##make predictions
test_xgb_tune5_w <- predict(xgb_tune5_w, newdata=testing)
#performace measurment
postResample(test_xgb_tune5_w, testing$Volume)
#RMSE=133.7762395   Rsquared=0.9884249  MAE=83.1027451 #a little better

#Relative error
realErro_w<- (test_xgb_tune5_w -testing$Volume)/testing$Volume  # to calc %
realErro_w
mean(realErro_w) #relative error = 0.2356211 #better than before

#plot predicted verses real
palette=c("Black","Green")
plot(test_xgb_tune5_w,testing$Volume, col=palette[as.factor(testing$ProductType.PLNS)])  #good for lower values (volume <500)
abline(a=0,b=1,col="red")
legend(x="bottomright", legend = c("Other","PLNS"), col=palette, pch=1)
#Good model for the products we want to predict! 

testing$prediction <- test_xgb_tune5_w
ggplot(testing, aes(Volume, prediction))+
  geom_point(size=2)+
  geom_abline(intercept = 0, slope = 1,color="blue")+
  labs(x="Real Volume",
       y="Predicted Volume",
       title="Predicted Volume vs Real Volume")


#Add the price
#start checking 1000 trees
nrounds <- 1000
tune_grid_p <- expand.grid(
  nrounds = seq(from=200, to = nrounds, by=50), #from 200 trees to 1000 50 by 50
  eta= c(0.025,0.05,0.1,0.3), #learning rate
  max_depth=c(2,3,4,5,6),
  gamma=0,
  colsample_bytree=1,
  min_child_weight=1,
  subsample=1
)

control_train <- trainControl(method = "repeatedcv", number = 10, repeats = 3) 
set.seed(123)
xgb_tune_p <- train(Volume~x4StarReviews+PositiveServiceReview+ShippingWeight+Price, data=training,
                    method="xgbTree",
                    trControl=control_train,
                    tuneGrid=tune_grid_p)
plot(xgb_tune_p)
xgb_tune_p$bestTune
# best:nrounds = 200, max_depth = 3, eta = 0.1, gamma =0, 
# colsample_bytree = 1, min_child_weight = 1 and subsample = 1

#find max depth and minimum child weight
tune_grid2_p <- expand.grid(
  nrounds = seq(from=50, to = nrounds, by=50), #from 50 trees to 1000 50 by 50
  eta= xgb_tune_p$bestTune$eta, #best learning rate previous modelling
  max_depth=(xgb_tune_p$bestTune$max_depth - 1:xgb_tune_p$bestTune$max_depth + 1), 
  gamma=0,
  colsample_bytree=1,
  min_child_weight=c(1,2,3),
  subsample=1
)
set.seed(123)
xgb_tune2_p <- train(Volume~x4StarReviews+PositiveServiceReview+ShippingWeight+Price, data=training,
                     method="xgbTree",
                     trControl=control_train,
                     tuneGrid=tune_grid2_p)
plot(xgb_tune2_p)
xgb_tune2_p$bestTune
#best: nrounds = 550, max_depth = 1, eta = 0.1, gamma =0, 
#colsample_bytree = 1, min_child_weight = 1 and subsample = 1.

#fix the child weigth 1 and max depth 1.
# find row and column sampling
tune_grid3_p <- expand.grid(
  nrounds = seq(from=50, to = nrounds, by=50), #from 50 trees to 1000 50 by 50
  eta= xgb_tune_p$bestTune$eta, #best learning rate previous modelling
  max_depth=xgb_tune2_p$bestTune$max_depth, 
  gamma=0,
  colsample_bytree=c(0.4,0.6,0.8,1.0),
  min_child_weight=xgb_tune2_p$bestTune$min_child_weight,
  subsample=c(0.5,0.75,1.0)
)
set.seed(123)
xgb_tune3_p <- train(Volume~x4StarReviews+PositiveServiceReview+ShippingWeight+Price, data=training,
                     method="xgbTree",
                     trControl=control_train,
                     tuneGrid=tune_grid3_p)
plot(xgb_tune3_p)
xgb_tune3_p$bestTune
#row =1 and column =.6
#fix the gamma
tune_grid4_p <- expand.grid(
  nrounds = seq(from=50, to = nrounds, by=50), #from 50 trees to 1000 50 by 50
  eta= xgb_tune_p$bestTune$eta, #best learning rate previous modelling
  max_depth=xgb_tune2_p$bestTune$max_depth, 
  gamma=c(0,0.05,0.1,0.5,0.7,0.9,1.0),
  colsample_bytree=xgb_tune3_p$bestTune$colsample_bytree,
  min_child_weight=xgb_tune2_p$bestTune$min_child_weight,
  subsample=xgb_tune3_p$bestTune$subsample
)
set.seed(123)
xgb_tune4_p <- train(Volume~x4StarReviews+PositiveServiceReview+ShippingWeight+Price, data=training,
                     method="xgbTree",
                     trControl=control_train,
                     tuneGrid=tune_grid4_p)
plot(xgb_tune4_p)
xgb_tune4_p$bestTune
#best gamma=0
# reduce learning rate
tune_grid5_p <- expand.grid(
  nrounds = seq(from=100, to = 10000, by=100), #from 100 trees to 10000 by 100
  eta= c(0.01,0.015,0.025,0.05,0.1), #best learning rate previous modelling
  max_depth=xgb_tune2_p$bestTune$max_depth, 
  gamma=xgb_tune4_p$bestTune$gamma,
  colsample_bytree=xgb_tune3_p$bestTune$colsample_bytree,
  min_child_weight=xgb_tune2_p$bestTune$min_child_weight,
  subsample=xgb_tune3_p$bestTune$subsample
)
set.seed(123)
xgb_tune5_p <- train(Volume~x4StarReviews+PositiveServiceReview+ShippingWeight+Price, data=training,
                     method="xgbTree",
                     trControl=control_train,
                     tuneGrid=tune_grid5_p)
plot(xgb_tune5_p)
xgb_tune5_p$bestTune
#nrounds = 1100, max_depth = 1, eta = 0.05, gamma =0, 
#colsample_bytree = 0.6, min_child_weight = 1 and subsample = 1.
#RMSE=216.7161 Rsquared=0.9249764 MAE:130.5042 

##make predictions
test_xgb_tune5_p <- predict(xgb_tune5_p, newdata=testing)
#performace measurment
postResample(test_xgb_tune5_p, testing$Volume)
#RMSE=133.7762395   Rsquared=0.9884249  MAE=83.1027451 #same as before

#Relative error
realErro_p<- (test_xgb_tune5_p -testing$Volume)/testing$Volume  # to calc %
realErro_p
mean(realErro_p) #relative error = 0.2356211 #same as before, price does not influence in the price.

#compare models
rValues2 <- resamples(list(xgbtree_2var=xgb_tune5,
                           xgbtree_3var=xgb_tune5_w,
                           xgbtree_4var=xgb_tune5_p))
rValues2$values
summary(rValues2)
bwplot(rValues2,metric="RMSE",
       main="xgbTree")





#Predictions on new products data set
#download data set
new <- read.csv("newproductattributes2017.2.csv",header=TRUE)
str(new)
summary(new)

#Predictions
predictions <- predict(xgb_tune5_w, newdata=new)

new$predictions <- predictions
write.csv(new, file="new_predictions.csv", row.names = TRUE)



#plot veriables of importance with plotly
plot_ly(x = existing_dum_PLNS$PositiveServiceReview, y = existing_dum_PLNS$x4StarReviews, 
        z = existing_dum_PLNS$Volume) %>% add_surface()

plot_ly(existing_out, x = ~x4StarReviews, y = ~PositiveServiceReview, z = ~Volume, color = ~ProductType) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = '4 Star Reviews'),
                      yaxis = list(title = 'Positive Service Reviews'),
                      zaxis = list(title = 'Sales Volume')))

plot_ly(existing_out, x = ~x4StarReviews, y = ~PositiveServiceReview, z = ~ShippingWeight, color = ~Volume) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = '4 Star Reviews'),
                      yaxis = list(title = 'Positive Service Reviews'),
                      zaxis = list(title = 'Shipping Weight')))


#code for hyperplane plot regression for two variables
x <- existing$x4StarReviews
y <- existing$PositiveServiceReview
z <- existing$Volume

fit <- lm(z ~ x + y)

grid.lines = 78
x.pred <- seq(min(x), max(x), length.out = grid.lines)
y.pred <- seq(min(y), max(y), length.out = grid.lines)
xy <- expand.grid( x = x.pred, y = y.pred)
z.pred <- matrix(predict(fit, newdata = xy), 
                 nrow = grid.lines, ncol = grid.lines)
# fitted points for droplines to surface
fitpoints <- predict(fit)
# scatter plot with regression plane, no interactive
scatter3D(x, y, z, pch = 18, cex = 2, bty = "g",
          theta = 30, phi = -20, ticktype = "detailed", #inclination
          xlab = "4STARS", ylab = "POSREV", zlab = "VOLUME",  
          surf = list(x = x.pred, y = y.pred, z = z.pred,  
          facets = NA, fit = fitpoints, main = "LM"))

surfaceplot<-plot_ly(existing,x=~x4StarReviews,y=~PositiveServiceReview,z=~Volume,
                     type="scatter3d",mode="marker")
surfaceplot <-add_trace(p=surfaceplot,z=z.pred,x=x.pred,y=y.pred,type="surface")
surfaceplot


ggplot(existing_out, aes(x4StarReviews,PositiveServiceReview,color=Volume))+
  geom_point()+
  labs(x="4 Star Reviews",
       y="Positive Service Review",
       title="Relationship of customer and service reviews and sales volume",
       color="Sales Volume")+
  coord_cartesian(xlim=c(0,100),ylim=c(100,0))

ggplot(existing_out, aes(Volume,ShippingWeight,color=Volume))+
  geom_point()+
  labs(x="Volume",
       y="Shipping Weight",
       title="Relationship of shipping weight and sales volume",
       color="Sales Volume")
