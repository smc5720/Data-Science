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
library(tm)
library(SnowballC)

sms_raw<-read.csv("D:/Study/Data Science/spam.csv", stringsAsFactors = FALSE)
sms_raw$type <- factor(sms_raw$type)
# 데이터를 불러올 떄, 문자메세지는 Factor로 받아오지 않도록하고
# label은 다시 factor로 변경해줍니다.
# ham과 spam의 비율을 확인합니다.
table(sms_raw$type)

# 코퍼스 객체생성 <- 벡터화 <- 텍스트 내용
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# tm_map 함수를 사용해서 각 corpus들에 함수를 적용 -> 소문자화
sms_corpus_clean <- tm_map(sms_corpus, tolower)

# 벡터에 문제가 생기는 경우가 간혹 있습니다.
# 그러한 경우에는 `content_transformer`를 적용합니다.
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

# 숫자 제거
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

# 구두점 
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

# stop words
stopwords()
sms_corpus_clean <- tm_map(sms_corpus_clean,removeWords, stopwords())

# 형태소 분석
sms_corpus_clean <- tm_map(sms_corpus_clean,stemDocument)

# 추가 여백 제거
sms_corpus_clean <- tm_map(sms_corpus_clean,stripWhitespace)

# 단어 토큰화
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm <- DocumentTermMatrix(sms_corpus,
                              control=list(
                                tolower = TRUE,
                                removeNumbers = TRUE,
                                stopwords = TRUE,
                                removePunctuation = TRUE,
                                stemming = TRUE))

# 특수문자 제거용 함수
replacePunctuation <- function(x){
  gsub(pattern = "[[:punct:]]+", " ", x)
}

# 숫자 제거용 함수
replaceNumber <- function(x){
  gsub(pattern = "[[:digit:]]+", " ", x)
}

# 만든 함수를 적용할 떄는 content_transformer 적용하는 것이 좋음
# 데이터에 적용
sms_corpus_clean <- tm_map(sms_corpus_clean, content_transformer(replacePunctuation))
sms_corpus_clean <- tm_map(sms_corpus_clean, content_transformer(replaceNumber))

# 나머지 전처리는 토큰화 할 때, 옵션으로 적용
sms_dtm<-DocumentTermMatrix(sms_corpus_clean,
                            control=list(
                              tolower = TRUE,
                              stopwords = TRUE,
                              stemming = TRUE
                            ))

# train data와 test 데이터 나누기
sms_dtm_train <- sms_dtm[1:4167,]
sms_dtm_test <- sms_dtm[4167:5559,]

sms_train_labels <- sms_raw[1:4167,]$type
sms_test_labels <- sms_raw[4167:5559,]$type

# 5번 이상 나온 feature 선택
sms_freq_words <- findFreqTerms(sms_dtm_train,5)

# 해당 feature에 해당되는 열만 선택
sms_dtm_freq_train <- sms_dtm_train[,sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[,sms_freq_words]

# 범주형으로 바꾸기위한 함수 생성
convert_counts<- function(x){ x <- ifelse(x>0, 'YES', "NO")}

# MARGIN=2 열을 기준으로 함수 적용
sms_train <- apply(sms_dtm_freq_train,MARGIN = 2,FUN = convert_counts)
sms_test <- apply(sms_dtm_freq_test,MARGIN = 2,FUN = convert_counts)

# 필터 모델 생성
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

sms_test_pred <- predict(sms_classifier, sms_test)

CrossTable(sms_test_pred, sms_test_labels,
           prop.t=FALSE, prop.r = FALSE,
           dnn=c('predicted','actual'))
# 예측 점수 확인
sum(sms_test_pred == sms_test_labels)*100/length(sms_test_labels)

