# clone related directories
git clone https://github.com/balancap/SSD-Tensorflow.git
git clone https://github.com/DrSleep/tensorflow-deeplab-resnet.git

# install requirements
pip install -r requirements.txt --upgrade

mkdir -p checkpoints
mkdir -p data/train
mkdir -p data/valid
mkdir -p data/test