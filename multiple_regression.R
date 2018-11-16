#Task 2.3: Multiple regression in R
#Lara Cobler Moncunill
#November 15th, 2018

library(readr)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

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

#Product number: no sense in  the analysis
existing_out$ProductNum <- NULL

#Build the correlation matrix
corr_existing_out <- cor(existing_out[,2:16]) #no include product type
corr_existing_out
#nice plot of the correlation matrix
corrplot(corr_existing_out,order="hclust",type="upper")

#remove variables that correlate between them (>0.5, <-0.5)
#5star,profit margin, 3star, 2star, 1star, negative service review, width and height
existing_out$x5StarReviews <- NULL
existing_out_less <- subset(existing_out,select=-c(x3StarReviews,x2StarReviews,x1StarReviews,
                                                     NegativeServiceReview,ProductWidth,
                                                   ProfitMargin,ProductHeight))
#Anova to see if the product type is related to volume
anova_res <- aov(Volume~ProductType,data=existing_out)
summary(anova_res)
#Yes it is.

#variable importance of the left
set.seed(333)
fit_rf <- randomForest(Volume~.,data=existing_out_less)
(VI_F=importance(fit_rf))
varImpPlot(fit_rf,type=2)
#oreder: positive service, 4x, product type, shipping w, product depth, price, recommend

#dummify the data, convert all factor or 'chr' classes to binary features
existing_dummy <- dummyVars("~.",data=existing_out_less)
existing_dum <- data.frame(predict(existing_dummy,newdata = existing_out_less))
str(existing_dum)

#variable importance with dummies
set.seed(333)
fit_rf_d <- randomForest(Volume~.,data=existing_dum)
(VI_F=importance(fit_rf_d))
varImpPlot(fit_rf_d,type=2)
#First the reviews, last product types separate, finally smartphones,PC,laptop, netbook

#do one dummy column wit PC, laptops, notebooks, and smartphones
existing_dum <- transform(existing_dum,ProductType.PLNS=ProductType.Laptop+ProductType.Netbook+ProductType.PC+ProductType.Smartphone)
#remove other dummies of the data frame
existing_dum_PLNS <- existing_dum[,13:20]

#annova
anova_PLNS <- aov(Volume~ProductType.PLNS,data=existing_dum_PLNS)
summary(anova_PLNS) #almost statistically significant

boxplot(existing_dum_PLNS$ProductType.PLNS,existing_dum_PLNS$Volume)

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

