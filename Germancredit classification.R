#Load Packages
if (!require('MASS')) install.packages('MASS')
if (!require('nnet')) install.packages("nnet")
if (!require('ggplot2')) install.packages('ggplot2')
if (!require('caret')) install.packages('caret')
if (!require('caTools')) install.packages('caTools')

library(MASS)
library(caret)  # for model training and fine-tuning
library(nnet)
library(ggplot2)
library(caTools)
library(tidyverse)
##for ROC 
###Read the data set, and relabel its columns
germancredit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
colnames(germancredit) <- c("chkactstatus", "duration", "credithistory", "purpose", "amount", "saving", "presentjob", "installmentrate", "sexstatus", "otherdebtor", "resident", "property", "age", "otherinstall", "housing", "ncredits", "job", "npeople", "telephone", "foreign", "response")
germancredit$response = as.factor(germancredit$response) # coded as 1=good_risk, 2=bad_risk
table(germancredit$response)
str(germancredit)
###Prep it
#The values for the class/response variable must be factors and must be valid names
germancredit$response <-as.factor(ifelse(germancredit$response==1, "Good", "Bad")) #recode response variable
#as coded, the default reference/first level is the "lowest" alphabetical value; here it is "Bad"
#Caret's default "Positive" class is the first level of the outcome variable in binary classification
#Make "Good" the first level
germancredit$response <- relevel(germancredit$response, "Good")
table(germancredit$response)
#divide into train and test
set.seed(123) #keep this for reproducability
in.train <- createDataPartition(germancredit$response, p=0.75, list=FALSE)
# 75% for training, one split
germancredit.train <- germancredit[in.train,]; dim(germancredit.train); table(germancredit.train$response)
germancredit.test <- germancredit[-in.train,];dim(germancredit.test); table(germancredit.test$response)
cmetric<- "ROC"
###Build a logistic regression model
set.seed(123)
logist <- train(response ~., data = germancredit.train, method = "glm", family = "binomial" , metric=cmetric,
                trControl = trainControl("cv", number = 10, summaryFunction=twoClassSummary, classProbs = TRUE)
)
logist
# Model coefficients and confusion matrix
coef(logist$finalModel) 
confusionMatrix(logist) 
# Make predictions and clauclate confusionmatrix on Test data
probsTest <- predict(logist, type="prob", newdata=germancredit.test)
#table(test.data$bmedv, probsTest[,2]>0.5)
threshold <- 0.5
pred      <- factor( ifelse(probsTest[, "Good"] > threshold, "Good", "Bad") )
confusionMatrix(pred, germancredit.test$response)
predslogist <- probsTest[,2]  ##save for later

# Build ElasticNet regression model
set.seed(123)
elastic <- train(response ~., data = germancredit.train, method = "glmnet", family="binomial", metric=cmetric, tuneLength = 10,
                 trControl = trainControl("cv", number = 10, summaryFunction=twoClassSummary, classProbs = TRUE)
)
elastic 
plot(elastic)
# Model coefficients
coef(elastic$finalModel, elastic$bestTune$lambda)
# Make predictions and calculate confusionmatrix on Test data
probsTest <- predict(elastic, type="prob", newdata=germancredit.test)
threshold <- 0.5
pred      <- factor( ifelse(probsTest[, "Good"] > threshold, "Good", "Bad") )
confusionMatrix(pred, germancredit.test$response)

# or simply table(test.data$bmedv, probsTest[,2]>0.5)
predselastic <-probsTest[,2]  #save for later


###build and fine tune Tree model
set.seed(123)
tree <- train(response ~., data = germancredit.train, method = "rpart", metric=cmetric,
              trControl = trainControl("cv", number = 10, summaryFunction=twoClassSummary, classProbs = TRUE)
)
tree
plot(tree)
#Visualize the tree 
par(xpd = NA) # Avoid clipping the text in some device
plot(tree$finalModel)
text(tree$finalModel, digits = 3)

confusionMatrix(tree) 
# Make predictions and calculate confusionmatrix on Test data
probsTest <- predict(tree, type="prob", newdata=germancredit.test)
threshold <- 0.5
pred      <- factor( ifelse(probsTest[, "Good"] > threshold, "Good", "Bad") )
confusionMatrix(pred, germancredit.test$response)
# or simply table(test.data$bmedv, probsTest[,2]>0.5)
predstree <- probsTest[,2]  ##save for later


#helper function adapted from: https://stackoverflow.com/questions/55936315/decision-boundary-plots-in-ggplot2

library(scales)
decisionplot <- function(model, data, class = NULL, predict_type = NULL,
                         resolution = 250, showgrid = TRUE, ftitle = NULL, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[, -which(names(data) %in% c(class))]
  k <- length(unique(cl))
  
  plot(data, col = c("red", "blue")[cl], pch = 20, ...)
  
  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  if(is.null(predict_type)) p <- predict(model, g) else p <- predict(model, g, type = predict_type)
  #if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = c("red", "blue")[p], pch = ".")
  
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  if(!is.null(ftitle)) title(main = ftitle) 
  
  invisible(z)
}

# Build Neural Network regression model
set.seed(123)
NNetwork <- train(response ~., data = germancredit.train, method="nnet", preProcess = "range", tuneLength=5, maxit=1000, linout=FALSE, metric=cmetric,
                  trControl = trainControl("cv", number = 10, summaryFunction=twoClassSummary, classProbs = TRUE)
)
NNetwork 
plot(NNetwork)

# Make predictions and calculate confusionmatrix on Test data
probsTest <- predict(NNetwork, type="prob", newdata=germancredit.test)
threshold <- 0.5
pred      <- factor( ifelse(probsTest[, "Good"] > threshold, "Good", "Bad") )
confusionMatrix(pred, germancredit.test$response)
# or simply table(test.data$bmedv, probsTest[,2]>0.5)
predsNNetwork <-probsTest[,2]  #save for later

#Plot the ROC curveCalculate the AUROC
colAUC(probsTest[,2], germancredit.test$response, plotROC=TRUE)

#compare the four models based on resampling
models <- list(logistic=logist, elastic = elastic, tree=tree, NNetwork=NNetwork)
results <- resamples(models)
summary(results)
summary(results, metric="ROC")
bwplot(results, metric="ROC")
compare_models(tree, elastic)
compare_models(tree, logist)
compare_models(logist, elastic)
compare_models(logist, NNetwork)
compare_models(NNetwork, elastic)
compare_models(tree, NNetwork)
summary(diff(results))

##compare the four models on the hold-out/Test partition
colAUC(cbind(predslogist,predstree, predselastic,predsNNetwork), germancredit.test$response, plotROC=TRUE)


