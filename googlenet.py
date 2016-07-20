import caffe
import os
import glob
import numpy as np

def main():
    model_trained = "data/caffe/bvlc_googlenet.caffemodel"
    model_prototxt = "data/caffe/deploy.prototxt"
    layer_name = "pool5/7x7_s1"
    input_dir = "data/transform/06/"
    output_dir = "data/caffe/06/"
    
    caffe.set_mode_cpu()
    net = caffe.Classifier(model_prototxt,
                        model_trained,
                        channel_swap=(2,1,0),
                        raw_scale=255,
                        image_dims=(256, 256))
    
    im_dirs = os.listdir(input_dir)
    
    print "Generating Features ..."
    for im_i, im_dir in enumerate(im_dirs):
        print "%d/%d" % (im_i + 1, len(im_dirs))

        in_dir = input_dir + im_dir + "/"
        
        features = []
        for im_file in glob.glob(in_dir + "*.png"):
            im_data = caffe.io.load_image(im_file)
            predict = net.predict([im_data], oversample = False)
            features.append(net.blobs[layer_name].data[0].reshape(1,-1))

        features = np.array(features)
        features = features.reshape(features.shape[1], features.shape[2])
        out_dir = output_dir + im_dir + "/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            
        np.savetxt(out_dir + "features.csv", features, delimiter = ",", fmt="%.8f")
        
        value = np.loadtxt(in_dir + "value.csv", delimiter = ",")
        np.savetxt(out_dir + "value.csv", [value], delimiter = ",")
if __name__ == "__main__":
    main()