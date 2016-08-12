require(caret)
features <- read.csv("data/features/06/features.csv", header = FALSE)
x <- features[,1:ncol(features) - 1]
y <- features[,ncol(features)]
set.seed(666)
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
grid <- expand.grid(C=exp(-10:10) / 100)
model <- train(x, y, trControl = train_control, method="svmLinear", metric = "Rsquared", preProc = c("center", "scale"), tuneGrid = grid)
print(model)
dir.create("data/r/")
dir.create("data/r/06/")
save(model, file = "data/r/06/model.RData")