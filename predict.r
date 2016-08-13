load(paste(data_path, "model.Rdata", sep=""))
features <- read.csv(paste(feature_path, "features.csv", sep=""), header = FALSE)
x <- features[1:nrow(features),1:ncol(features) - 1]
y <- features[1:nrow(features),ncol(features)]
prediction <- predict(model, x)
jpeg(paste(data_path, "residuals.jpg", sep=""))
plot(y, prediction, ylab="Residuals", xlab = "Larvae Count", main="Fly Habitats")
abline(0, 0)
dev.off()