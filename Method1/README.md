## Simulium-Damnosum
Code for simulium damnosum models

## Setup
###Install Miniconda (for all users): http://conda.pydata.org/miniconda.html
###Install the following conda libraries
* `conda install scipy`
* `conda install scikit-learn`
* `conda install gdal`
* `conda install -c menpo opencv=2.4.11`

###Other Libraries
* `pip install protobuf`
* [install tensorflow](https://www.tensorflow.org/install/)

##Usage
####Put a .TIF file and a .csv file in `../data/sat/06`
####The csv file needs 4 columns: ID/Longitude/Latitude/Dependent_Variable
####Run python files in order:
* extract_images.py
* resnet.py
* svr.py

####svr.py generates kfold split results as an html table