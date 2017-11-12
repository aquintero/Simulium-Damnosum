import os
import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

def main():
    input_dir = "data/sites/06/"
    output_dir = "data/features/06/"
    img_dirs = os.listdir(input_dir)
    
    img_shape = (222, 222, 3)
    resnet = ResNet50(include_top = False, weights = 'imagenet', input_shape = img_shape)
    
    print("Generating Features ...")
    
    features = []
    for img_file in glob.glob(input_dir + "*.png"):
        x = image.load_img(img_file, target_size = img_shape[:2])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        features.append(resnet.predict(x))

    features = np.array(features)
    features = np.squeeze(features)
    print(features.shape)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    np.savetxt(output_dir + "features.csv", features, delimiter = ",", fmt="%.8f")
    
    values = np.loadtxt(input_dir + "values.csv", delimiter = ",")
    np.savetxt(output_dir + "values.csv", values, delimiter = ",")
if __name__ == "__main__":
    main()