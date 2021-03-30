Example code for "Drift correction in localization microscopy using entropy minimization"
-----------------------------------------------------------------------------------------

To run this, currently a Windows 64-bit PC is required with CUDA 10.2 installed. 
The algorithm is able to run on CPU only - we're working on removing the CUDA dependency from the code and making a standalone demo.

Step 1. 
Install python. Anaconda is recommended: https://www.anaconda.com/distribution/

Step 2.
Create a virtual environment, such as an anaconda environment:

conda create -n myenv anaconda python=3.8
conda activate myenv

Step 3.
Install photonpy:

pip install photonpy

Step 4.  Run the example code:
python drift-estimation-example.py
