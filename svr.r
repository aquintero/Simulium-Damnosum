require(caret)
features <- read.csv("data/features/06/features.csv", header = FALSE)
x <- features[,1:ncol(features) - 1]
y <- features[,ncol(features)]
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
model <- train(x, y, trControl = train_control, method="svmRadial", preProc = c("center", "scale"))
print(model)
dir.create("data/r/")
dir.create("data/r/06/")
save(model, file = "data/r/06/model.RData")