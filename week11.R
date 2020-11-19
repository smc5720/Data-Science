library(arules)
library(arulesViz)
library(wordcloud)

FOREIGN <- read.csv("D:/Study/Data Science/tour/tourdata.csv")
Code <- read.csv("D:/Study/Data Science/tour/Code.csv")

ITEM = c()

for(k in 1:ncol(FOREIGN)){
  ITEM = c(ITEM, FOREIGN[, k])
}

ID = rep(1:nrow(FOREIGN), ncol(FOREIGN))

ITEM_LIST = data.frame(
  ID = ID,
  ITEM = ITEM
)

ITEM_LIST = na.omit(ITEM_LIST)

ITEM_LIST = merge(ITEM_LIST, Code, by.x = "ITEM", by.y = "No", all = TRUE)
ITEM_LIST = na.omit(ITEM_LIST)

item.list = split(ITEM_LIST$Name, ITEM_LIST$ID)
item.transaction = as(item.list, "transactions")
item.matrix = as(item.transaction,'matrix')

Sums = sort(colSums(item.matrix),decreasing = TRUE)
item.matrix = item.matrix[, names(Sums)]

rules_0.15 = apriori(item.transaction,
                     parameter = list(support = 0.05,
                                      confidence = 0.1,
                                      minlen = 2)
                     )

inspect(rules_0.15)

rules_conf <- sort(rules_0.15, by="confidence", decreasing=TRUE)
inspect(rules_conf)

rules_lift <- sort (rules_0.15, by="lift", decreasing=TRUE)
inspect(rules_lift)

