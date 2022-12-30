#!/bin/bash
FILE=/packages-microsoft-prod.deb
if test -f "$FILE"; then
   	sudo dpkg -i FILE
else
	wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
	sudo dpkg -i FILE
fi
# linux package installs
sudo apt-get update
sudo apt-get install fuse -y
sudo apt-get install ffmpeg -y
sudo apt-get install libsndfile1 -y


# python package installs
pip install -U fusepy --user
pip install azureml azureml-core

# tensorflow
pip uninstall tensorflow -y
pip install tensorflow==2.4.1
pip install keras==2.4.3

# datasets
pip uninstall datasets -y
pip install torch torchaudio librosa datasets[audio]

# transformers
conda uninstall tokenizers,

#ctc
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install pyctcdecode

