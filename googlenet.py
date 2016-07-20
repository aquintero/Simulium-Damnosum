import caffe
import os

def main():
    model_prototxt = "data/caffe/bvlc_alexnet.caffemodel"
    model_trained = "data/caffe/deploy.prototxt"
    layer_name = "fc6"
    input_dir = "data/transform/06/"
    output_dir = "data/overfeat/06/"
    
    caffe.set_mode_cpu()
    net = caffe.Classifier(model_prototxt,
                        model_trained,
                        caffe.TEST)
    
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
        out_dir = output_dir + im_dir + "/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            
        np.savetxt(out_dir + "features.csv", features, delimiter = ",")
        
        value = np.loadtxt(in_dir + "value.csv", delimiter = ",")
        np.savetxt(out_dir + "value.csv", [value], delimiter = ",")
if __name__ == "__main__":
    main()