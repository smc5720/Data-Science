library(readxl)
library(gmodels)
library(rpart.plot)
library(rattle)
library(RColorBrewer)
library(doBy)
library(sampling)
library(caret)
library(rpart)
library(e1071)

df <- read.csv(file="D:/Study/Data Science/titanic.csv", header=T)
head(df)

train.control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = TRUE
)

df$survived <- as.factor(df$survived)

set.seed(1)
rpartFit1 <- train(survived ~ pclass + sex + age + fare,
                   data = df,
                   method = "rpart2",
                   tuneLength = 6,
                   trControl = train.control,
                   metric = "ROC"
)

plot(rpartFit1)
fancyRpartPlot(rpartFit1$finalModel)

pre_prune_tree <- rpart(survived ~ pclass + sex + age + fare,
                        data = df,
                        maxdepth = 12)

fancyRpartPlot(pre_prune_tree)

tree <- rpart(survived ~ pclass + sex + age + fare,
              data = df,
              cp = 0,
              minbucket = 1
)

printcp(tree)
plotcp(tree)

fancyRpartPlot(tree)

post_prune_tree <- prune(tree, cp =0.01)

fancyRpartPlot(post_prune_tree)


rpartpred<-predict(pre_prune_tree, df, type='class')
confusionMatrix(table(rpartpred, df$survived))

rpartpred<-predict(post_prune_tree, df, type='class')
confusionMatrix(table(rpartpred, df$survived))

str(pre_prune_tree)

rpart.rules(pre_prune_tree)

x <- df$survived

result <- data.frame(Real = x, Predict = rpartpred)
CrossTable(result$Real, result$Predict)
