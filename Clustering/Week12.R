dust_origin <- read.csv("D:/Study/Data Science/dust_day.csv", stringsAsFactors = TRUE)

dust_df <- dust_origin[-c(1:2)] # �ʿ���� ���� ����

region <- c("������", "������", "���ϱ�", "������",
            "���Ǳ�", "������", "���α�", "��õ��",
            "�����", "������", "���빮��", "���۱�",
            "������", "���빮��", "���ʱ�", "������",
            "���ϱ�", "���ı�", "��õ��", "��������",
            "��걸", "����", "���α�", "�߱�", "�߶���")

mean.dust <- c(0)
mean.minidust <- c(0)
mean.ozone <- c(0)
mean.nitro <- c(0)
mean.mono <- c(0)
mean.gas <- c(0)

for(i in 1:25)
{
  dust_region <- dust_df[dust_df$�����Ҹ�==region[i], ]
  mean.dust[i] <- mean(dust_region$�̼�����.....)
  mean.minidust[i] <- mean(dust_region$�ʹ̼�����.....)
  mean.ozone[i] <- mean(dust_region$����.ppm.)
  mean.nitro[i] <- mean(dust_region$�̻�ȭ���ҳ�.ppm.)
  mean.mono[i] <- mean(dust_region$�ϻ�ȭź�ҳ�.ppm.)
  mean.gas[i] <- mean(dust_region$��Ȳ�갡����.ppm.)
}

region_df <- data.frame(region, mean.dust, mean.minidust, mean.ozone, mean.nitro, mean.mono, mean.gas)

# ����ȭ
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

# kmeans ���
library(stats)

# test �����͸� 5���� ����ȭ
kmeans.result <- kmeans(dust_n,5)

kmeans.result$tot.withinss
kmeans.result$betweenss
print(100 * kmeans.result$betweenss/kmeans.result$totss)

kmeans.result

# �ð�ȭ
library(factoextra)
fviz_cluster(kmeans.result,data=dust_n,stand=T)

library(ggplot2)
result <- cbind(dust_n, kmeans.result$cluster)

ggplot(data=result,
       mapping = aes(x=mean.dust, y=mean.minidust,
                     color=as.factor(kmeans.result$cluster))) + geom_point() + geom_text(aes(label=row.names(result)),hjust=0, vjust=0)