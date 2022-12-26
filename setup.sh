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
    apt install ffmpeg -y
else
    echo "Please run with sudo"
fi

python3.7 -m venv py37
source py37/bin/activate
pip install -U pip
pip3 install -r requirements.txt
pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install kenlm-master.zip
pip3 install pyctcdecode==v0.1.0 --target=/py37/pyctcdecode01
pip3 install pyctcdecode==v0.4.0 --target=/py37/pyctcdecode04
