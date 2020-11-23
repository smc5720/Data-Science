dust_origin <- read.csv("D:/Study/Data Science/dust_day.csv", stringsAsFactors = TRUE)

dust_df <- dust_origin[-c(1:2)] # 필요없는 변수 제거

region <- c("강남구", "강동구", "강북구", "강서구",
            "관악구", "광진구", "구로구", "금천구",
            "노원구", "도봉구", "동대문구", "동작구",
            "마포구", "서대문구", "서초구", "성동구",
            "성북구", "송파구", "양천구", "영등포구",
            "용산구", "은평구", "종로구", "중구", "중랑구")

mean.dust <- c(0)
mean.minidust <- c(0)
mean.ozone <- c(0)
mean.nitro <- c(0)
mean.mono <- c(0)
mean.gas <- c(0)

for(i in 1:25)
{
  dust_region <- dust_df[dust_df$측정소명==region[i], ]
  mean.dust[i] <- mean(dust_region$미세먼지.....)
  mean.minidust[i] <- mean(dust_region$초미세먼지.....)
  mean.ozone[i] <- mean(dust_region$오존.ppm.)
  mean.nitro[i] <- mean(dust_region$이산화질소농도.ppm.)
  mean.mono[i] <- mean(dust_region$일산화탄소농도.ppm.)
  mean.gas[i] <- mean(dust_region$아황산가스농도.ppm.)
}

region_df <- data.frame(region, mean.dust, mean.minidust, mean.ozone, mean.nitro, mean.mono, mean.gas)

# 정규화
normalize <- function(x) {
  return ((x-min(x))/(max(x)-min(x)))
}

dust_n <- as.data.frame(lapply(region_df[2:3], normalize))
rownames(dust_n) = region

dust_confirm <- dust_n
dust_confirm[, "sum"] <- rowSums(dust_n)

hc = hclust(dist(dust_n))
plot(hc)

rect.hclust(hc, k = 5, border = 2:7)

cutree(hc, k = 5)

# kmeans 사용
library(stats)

# test 데이터를 5개로 군집화
kmeans.result <- kmeans(dust_n,5)

kmeans.result$tot.withinss
kmeans.result$betweenss
print(100 * kmeans.result$betweenss/kmeans.result$totss)

kmeans.result

# 시각화
library(factoextra)
fviz_cluster(kmeans.result,data=dust_n,stand=T)

library(ggplot2)
result <- cbind(dust_n, kmeans.result$cluster)

ggplot(data=result,
       mapping = aes(x=mean.dust, y=mean.minidust,
                     color=as.factor(kmeans.result$cluster))) + geom_point() + geom_text(aes(label=row.names(result)),hjust=0, vjust=0)
