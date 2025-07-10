#!/bin/bash
apt update && apt install -y [...] && apt clean
python -m pip install --upgrade pip

# First install huggingface_hub with the specific version
#pip install huggingface_hub==0.16.4

# Then install other packages with specific versions
#pip install datasets==2.13.0  # Specify a version compatible with huggingface_hub 0.16.4
#pip install accelerate==0.22.0
#pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
#pip install git+https://github.com/huggingface/transformers.git@c030fc891395d11249046e36b9e0219685b33399

# Install remaining requirements
pip install -r /home/banwari/llm_energy/synatra/evaluation/mind2web/requirements.txt