library(readxl)
library(dplyr)
library(caret)
library(rpart)
library(nnet)
library(neuralnet)
library(devtools)
library(reshape)
library(NeuralNetTools)
library(e1071)
library(pROC)
library(randomForest)
source_url('https://gist.githubusercontent.com/Peque/41a9e20d6687f2f3108d/raw/85e14f3a292e126f1454864427e3a189c2fe33f3/nnet_plot_update.r')

library(MASS)
library(psych)

Boston_df <- read.csv("D:/Study/Data Science/BostonHousing.csv", stringsAsFactors = TRUE)

#pairs.panels(Boston_df[, 1:14], method = "pearson", hist.col = "#00AFBB", density = TRUE, ellipses = TRUE)

# Data Partition
indexes = createDataPartition(Boston_df$MEDV, p = 0.7, list = F)
train = Boston_df[indexes, ]
test = Boston_df[-indexes, ]

control_lm <-trainControl(method = "repeatedcv",
                          number=10, repeats=5)

# ¼³¸í ¸ðµ¨
model_explain <- train(MEDV~., data=Boston_df,
               method="lm", trControl=control_lm,
               tuneLength=5, metric = 'RMSE')

model_lm <- model_explain$finalModel

summary(model_lm)

model_lm$residuals

plot(density(model_lm$residuals), main="Density Plot: Residuals", ylab="Frequency", 
     sub=paste("Skewness:", round(e1071::skewness(model_lm$residuals), 2)))

full <- model_lm
null <- lm(MEDV ~ 1, data=Boston_df)

# Stepwise-selection
step(null, direction = "forward", scope=list(lower=null, upper=full))

# ¿¹Ãø ¸ðµ¨
model_predict <- train(MEDV~., data=train,
               method="lm", trControl=control_lm,
               tuneLength=5, metric = 'RMSE')

model_pd <- model_predict$finalModel

medvPred <- predict(model_pd, test)

summary(model_pd)

actuals_preds <- data.frame(cbind(actuals=test$MEDV, predicteds=medvPred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)
correlation_accuracy
head(actuals_preds)
# correlation between the actuals and predicted values can be used as 
# a form of accuracy measure

# MinMaxAccuracy and MeanAbsolutePercentageError (MAPE)
min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))
min_max_accuracy

mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)  
mape

# k- Fold Cross validation
library(DAAG)
windows()
cvResults <- suppressWarnings(
  CVlm(data = Boston_df, 
       form.lm=MEDV ~ .,
       m=5, 
       dots=FALSE, 
       seed=29, 
       legend.pos="topleft",  
       printit=FALSE, 
       main="Small symbols are predicted values while bigger ones are actuals."
       )
  );  
# performs the CV

attr(cvResults, 'ms')
