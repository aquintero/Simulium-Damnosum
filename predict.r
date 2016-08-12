require(logging)
load("data/r/06/model.Rdata")
features <- read.csv("data/features/06/features.csv", header = FALSE)
x <- features[15:nrow(features),1:ncol(features) - 1]
y <- features[15:nrow(features),ncol(features)]
prediction <- predict(model, x)
loginfo(model$residuals)
jpeg("data/r/06/residuals.jpg")
plot(y, model$residuals, ylab="Residuals", xlab = "Larvae Count", main="Fly Habitats")
abline(0, 0)
dev.off()