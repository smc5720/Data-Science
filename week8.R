library(readxl)
library(dplyr)
library(caret)
library(nnet)
library(neuralnet)
library(devtools)
library(reshape)
library(NeuralNetTools)
source_url('https://gist.githubusercontent.com/Peque/41a9e20d6687f2f3108d/raw/85e14f3a292e126f1454864427e3a189c2fe33f3/nnet_plot_update.r')

# 타이타닉 데이터셋 불러오기
df<-read.csv("D:/Study/Data Science/titanic_final.csv", stringsAsFactors = FALSE)
names(df)[1] <- c("pclass")

data(df)

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
set.seed(1)
indexes = createDataPartition(n_titanic_df$survived, p = .6, list = F)
train_titanic = n_titanic_df[indexes, ]
test_titanic = n_titanic_df[-indexes, ]

# repeatedcv 방식 사용하여 최적 파라미터 탐색
set.seed(4)
control <-trainControl(method = "repeatedcv", number=10, repeats=5)
model <- train(survived~., data=train_titanic, method="nnet", trControl=control, tuneLength=5)
print(model)
ggplot(model)


'''
lda_data<-learning_curve_dat(dat = train_titanic,
                             outcome = "survived",
                             test_prop = 0,
                             method = "nnet",
                             trControl = control)

ggplot(lda_data, aes(x=Training_Size, y=Accuracy, color = Data)) +
  geom_smooth(method=loess, span = .8) +
  theme_bw()
'''

# nnet 인공신경망 예측모델 작성(size = 1, decay = 0.001)
nnet_model1 <- nnet(survived ~ pclass.1+pclass.2+pclass.3+sexfemale+sexmale+age+fare,
                    data = train_titanic,
                    size=5,
                    decay = 0.001,
                    maxit=100)

nnet_model2 <- nnet(survived ~ pclass.1+pclass.2+pclass.3+sexfemale+sexmale+age+fare,
                    data = train_titanic,
                    size =5,
                    decay = 0.001,
                    maxit = 300)

nnet_model3 <- nnet(survived ~ pclass.1+pclass.2+pclass.3+sexfemale+sexmale+age+fare,
                    data = train_titanic,
                    size =5,
                    decay = 0.001,
                    maxit = 500)

nnet_model4 <- nnet(survived ~ pclass.1+pclass.2+pclass.3+sexfemale+sexmale+age+fare,
                    data = train_titanic,
                    size =5,
                    decay = 0.001,
                    maxit = 1000)


summary(nnet_model1)

plot.nnet(nnet_model1)

# 인공신경망으로 예측

pre_titanic1 <- predict(nnet_model1, test_titanic, type = "class")
actual1 <- test_titanic$survived
table(actual1, pre_titanic1)

pre_titanic2 <- predict(nnet_model2, test_titanic, type = "class")
actual2 <- test_titanic$survived
table(actual2, pre_titanic2)

pre_titanic3 <- predict(nnet_model3, test_titanic, type = "class")
actual3 <- test_titanic$survived
table(actual3, pre_titanic3)

pre_titanic4 <- predict(nnet_model4, test_titanic, type = "class")
actual4 <- test_titanic$survived
table(actual4, pre_titanic4)

#정확도

confusionMatrix(factor(pre_titanic1), test_titanic$survived)
confusionMatrix(factor(pre_titanic2), test_titanic$survived)
confusionMatrix(factor(pre_titanic3), test_titanic$survived)
confusionMatrix(factor(pre_titanic4), test_titanic$survived)

maxit_range = seq(0, 1000, 100)

learn_curve <-data.frame(maxit=double(length(maxit_range)),
                         accuracy = double(length(maxit_range)))
idx = 0
for(i in maxit_range) {
  learn_curve$maxit[idx] = i
  model <- train(survived~., data=train_titanic, method="nnet", metric = 'Accuracy', maxit=i)
  learn_curve$accuracy[idx] <- max(model$results$Accuracy)
  idx = idx + 1
}

learn_curve <- learn_curve[1:10,]
g<-ggplot(data=learn_curve)
g<-g+geom_line(mapping=aes(x=maxit,y=accuracy))
g[["labels"]] = list(x='Maxit', y='Accuracy')
g<-g+stat_smooth(data=learn_curve,mapping=aes(x=maxit,y=accuracy))
g
