#!/bin/bash
if [ "$UID" -eq 0 -o "$EUID" -eq 0 ]; then
    apt update
    apt install software-properties-common -y
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update
    apt install python3.7 -y
    apt install python3.7-dev -y
    apt install python3.7-venv -y
    apt install python3-pip -y
else
    echo "Please run with sudo"
fi

python3.7 -m venv s2t
source s2t/bin/activate
pip install -U pip
cd Wav2Vec
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio
pip3 install kenlm-master.zip
