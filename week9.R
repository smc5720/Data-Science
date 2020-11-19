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

set.seed(1)

# Data Preprocessing
ds <- read.csv("D:/Study/Data Science/titanic_final.csv", stringsAsFactors = TRUE)
names(ds)[1] <- c("pclass")

ds <- ds %>% mutate(survived = factor(survived), pclass = factor(pclass))
levels(ds$survived) <- c("dead", "survived")

# Missing values
#aggr(ds, prop=FALSE, combined=TRUE, numbers=TRUE, sortVars=TRUE, sortCombs=TRUE)
ds <- ds %>% filter(!is.na(fare))
ds <- ds %>% filter(!is.na(age))

full <- data.frame(survived = ds$survived,
                   pclass = ds$pclass,
                   sex = ds$sex,
                   age = ds$age,
                   fare = ds$fare
)

# Data Partition
indexes = createDataPartition(full$survived, p = 0.7, list = F)
train = full[indexes, ]
test = full[-indexes, ]

# Decision Tree
# full tree (base model)
set.seed(1)
basemod <- rpart(survived~., 
                 data=train, 
                 cp = -1, 
                 minsplit = 2,
                 minbucket = 1)

# post pruning
bestcp <- basemod$cptable[which.min(basemod$cptable[,"xerror"]), "CP"]
postpr <- prune(basemod, bestcp)

Tree_pred <- predict(postpr, newdata=test, type="prob")

# Regression
set.seed(1)
GLM <- step(glm(survived~., data=train, family=binomial(link="logit")))
GLM_pred <- predict(GLM, newdata = test, type="response")




# 타이타닉 데이터셋 불러오기
df<-read.csv("D:/Study/Data Science/titanic_final.csv")
names(df)[1] <- c("pclass")

# 필요한 변수로만 데이터 프레임 재구성하기
titanic = data.frame(survived = df$survived,
                     sex = df$sex,
                     pclass = factor(df$pclass),
                     age = df$age,
                     fare = df$fare)
#원 핫 인코딩
dmy <- dummyVars(~., data = titanic)
titanic_df <- data.frame(predict(dmy, newdata = titanic))

# survived 항목 factor로 변경
titanic_df$survived <- factor(titanic_df$survived)

glimpse(titanic_df)
summary(titanic_df)
head(titanic_df)

#결측치 제거
row_na = apply(titanic_df,1,anyNA)
titanic_df<-titanic_df[!row_na,]
summary(titanic_df)

#정규화
n_titanic = preProcess(titanic_df, method = "range")
n_titanic_df = predict(n_titanic, titanic_df)
summary(n_titanic_df)
head(n_titanic_df)

# 테스트셋, 트레인셋 분리
indexes = createDataPartition(n_titanic_df$survived, p = .6, list = F)
train_titanic = n_titanic_df[indexes, ]
test_titanic = n_titanic_df[-indexes, ]

# repeatedcv 방식 사용하여 최적 파라미터 탐색
set.seed(1)
control <-trainControl(method = "repeatedcv", number=10, repeats=5)
model <- train(survived~., data=train_titanic, method="nnet", trControl=control, tuneLength=5)
print(model)
ggplot(model)


# nnet 인공신경망 예측모델 작성(size = 9, decay = 0.001)
set.seed(1)
nnet_model <- nnet(survived ~ pclass.1+pclass.2+pclass.3+sexfemale+sexmale+age+fare,
                   data = train_titanic,
                   size=9,
                   decay = 0.001,
                   maxit=1000)


summary(nnet_model)

plot.nnet(nnet_model)

# 인공신경망으로 예측

pre_titanic <- predict(nnet_model, test_titanic, type = "class")
actual <- test_titanic$survived
table(actual, pre_titanic)

#정확도

confusionMatrix(factor(pre_titanic), test_titanic$survived)

pre_titanic <- predict(nnet_model, n_titanic_df, type = "raw")

require(randomForest)
require(mlbench)

control_rf <-trainControl(method = "repeatedcv",
                          number=10, repeats=5)

model_rf <- train(survived~., data=n_titanic_df,
                  method="rf", trControl=control_rf,
                  tuneLength=5, metric = 'Accuracy')

model_rf$finalModel

pred_rf <- predict(model_rf$finalModel, newdata = n_titanic_df, type = "vote") 
pred_rf
roc_rf<-roc(n_titanic_df$survived, pred_rf[,1])

levels(n_titanic_df$survived)
plot(roc_rf)
auc(roc_rf)

titanic_rf<-randomForest(survived ~ ., data=n_titanic_df, ntree=100, importance = T)
titanic_rf

varImpPlot(titanic_rf)

#SVMLinear를 사용하여 최적 파라미터 탐색
set.seed(1)
tuneGrid = expand.grid(cost = 10**(-5:0))
tuneGrid
model <- train(survived~., data=train_titanic, method="svmLinear2", trControl=control, tuneGrid=tuneGrid)
print(model)
plot(model)

# svmLinear 예측모델 작성(cost = 0.01)
set.seed(1)
svm_model <- svm(survived ~ pclass.1+pclass.2+pclass.3+sexfemale+sexmale+age+fare,
                 data = train_titanic,
                 probability =TRUE,
                 kernel = "linear",
                 cost = 0.01
)
# svm model로 예측 
pre_svm <- predict(svm_model, test_titanic, type="raw", probability=TRUE)
actual <- test_titanic$survived
table(actual, pre_svm)

#정확도
confusionMatrix(pre_svm, test_titanic$survived)

pre_svm <- predict(svm_model, n_titanic_df, type = "raw", probability=TRUE)


# ROC curve
Tree_ROC <- roc(test$survived, Tree_pred[,2])
GLM_ROC <- roc(test$survived, GLM_pred)
rf.roc<-roc(n_titanic_df$survived, titanic_rf$votes[, 2])
nnet.roc<-roc(n_titanic_df$survived, pre_titanic[, 1])
svm.roc<-roc(n_titanic_df$survived, attr(pre_svm,'probabilities')[,2])

roc.test(Tree_ROC, GLM_ROC, rf.roc, nnet.roc, svm.roc, plot=TRUE)

plot.roc(Tree_ROC,
         col="red",
         print.auc=TRUE, 
         max.auc.polygon = TRUE,
         print.thres=TRUE, print.thres.pch=19, 
         print.thres.col="red", print.thres.adj=c(0, 2.8),
         auc.polygon = TRUE, auc.polygon.col="#D1F2EB")

plot.roc(GLM_ROC,
         add=TRUE,
         col="blue",
         print.auc=TRUE, print.auc.adj=c(0, -0.1),
         print.thres=TRUE, print.thres.pch=19, print.thres.col="blue")

plot.roc(roc_rf,
         add=TRUE,
         col="green",
         print.auc=TRUE, print.auc.adj=c(0, -1.1),
         print.thres=TRUE, print.thres.pch=19, print.thres.col = "green")

plot.roc(nnet.roc,
         add=TRUE,
         col="purple",
         print.auc=TRUE, print.auc.adj=c(1.11,1.2),
         print.thres=TRUE, print.thres.pch=19, print.thres.col = "purple",
         print.thres.adj=c(-0.085,1.1))

plot.roc(svm.roc,
         add=TRUE,
         col="yellow",
         print.auc=TRUE, print.auc.adj=c(1.11,2.5),
         print.thres=TRUE, print.thres.pch=19, print.thres.col = "yellow",
         print.thres.adj=c(-0.085,1.1))

auc(Tree_ROC)
auc(GLM_ROC)
auc(rf.roc)
auc(nnet.roc)
auc(svm.roc)

dev.off()