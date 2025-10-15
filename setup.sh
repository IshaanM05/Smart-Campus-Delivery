#!/bin/bash

# Activate virtual environment
source consultx_env/bin/activate

# Install requirements
pip install -r requirements.txt

echo "Setup complete. Run 'source consultx_env/bin/activate' to activate the env, then 'python main.py' to start the simulator."
#./setup.sh