library(caret)
features <- read.csv(paste(feature_path, "features.csv", sep=""), header = FALSE)
x <- features[,1:ncol(features) - 1]
y <- features[,ncol(features)]
set.seed(1)
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
grid <- expand.grid(C=exp(-10:10) / 100)
model <- train(x, y, trControl = train_control, method="svmLinear", metric = "Rsquared", preProc = c("center", "scale"), tuneGrid = grid)
print(model)
save(model, file = paste(data_path, "model.RData", sep=""))