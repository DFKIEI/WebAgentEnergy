#!/bin/bash
apt update ; apt install -y [...] ; apt clean
conda install [...]
python -m pip install --upgrade pip

pip install -r requirements.txt