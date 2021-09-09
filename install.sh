#!/bin/bash

# create libraries folder
mkdir lib
cd lib

# install pytorch DGCNN
git clone https://github.com/muhanzhang/pytorch_DGCNN
cd pytorch_DGCNN
cd lib
make -j4

# install SEAL
cd ../../
git clone https://github.com/muhanzhang/SEAL

# back to base directory
cd ..

# install python packages
pip install -r  requirements.txt

