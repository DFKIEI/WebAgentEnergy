#!/bin/bash
apt update && apt install -y [...] && apt clean
python -m pip install --upgrade pip
pip uninstall -y transformer-engine accelerate
pip install -r requirements.txt
pip install accelerate==0.22.0 --no-deps