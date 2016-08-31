## Simulium-Damnosum
Code for simulium damnosum models

## Setup
###Install Miniconda (for all users): http://conda.pydata.org/miniconda.html
###Install the following conda libraries
* `conda install PIL`
* `conda install scipy`
* `conda install scikit-image`
* `conda install scikit-learn`
* `conda install gdal`
* `conda install -c menpo opencv=2.4.11`
* `conda install -c r rpy2=2.8.2`

###Other Libraries
* `pip install protobuf`

###R
* Create environment variable R_HOME and set it to the R folder in your Miniconda installation
* Open the R shell and run `install.packages("caret")`
* run `install.packages("kernlab")

###Build Caffe: https://github.com/BVLC/caffe/tree/windows
* `git clone -b windows https://github.com/BVLC/caffe.git`
* Rename `windows/CommonSettings.props.example` to `CommonSettings.props`
* Open `CommonSettings.props` and set `PythonSupport` to `true`, `UseCuDNN` to `false` and `PythonDir` to your Miniconda installation
* Install Visual Studio 2013: https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx
* Build windows/caffe.sln for Release x64
* Go back to the root folder of caffe and copy `Build/x64/Release/pycaffe/caffe` to your miniconda installation at `Lib/site-packages`
* In the root caffe folder run `python scripts/download_model_binary.py models/bvlc_googlenet`
* From `models/bvlc_googlenet` copy `deploy.prototxt` and `bvlc_googlenet.caffemodel` to `Simulium-Damnosum/data/cafe`

##Usage
####Set the resolution in config.cfg
####Put a .TIF file and a .csv file in `Simulium-Damnosum/data/sat/<resolution>`
####The csv file needs 4 columns: ID/Longitude/Latitude/Dependent_Variable
####Run python files in order:
* extract_images.py
* transform.py
* googlenet.py
* group_features.py
* svr.py
####svr.py displays a psuedo r squared score of the regression model