Example code for "Drift correction in localization microscopy using entropy minimization"
-----------------------------------------------------------------------------------------

Article link:
https://www.biorxiv.org/content/10.1101/2021.03.30.437682v1

To run dme_example.py using pre-build Windows binaries:
-------------------------------------------------------

1. Install python. Anaconda is recommended: https://www.anaconda.com/distribution/
2. Create a virtual environment, such as an anaconda environment:

```
conda create -n myenv anaconda python=3.8
conda activate myenv
```

3. Install required pip packages, installing these within the virtual environment:

```
pip install tqdm scipy numpy matplotlib 
```

4.  Run the example code:

```
python dme_example.py
```

Build from source
-----------------
- On Windows, make sure to install CUDA 11.2 and build using dme/DriftEstimation.sln. Make sure to build in Release mode
- On Linux (tested on Ubuntu 20.04):
  - Install CUDA from the nVidia website. Install cmake and use "cmake .", followed by "make" in the repository root. 
  - This should build ./dme/bin/libdme.so
  - dme_example.py should now run without errors. 
- On MacOS, CUDA is no longer supported by nVidia, so only the CPU version can be used. 
  - Thanks to Duncan Ryan for help getting the GCC build working and confirming it runs on MacOS!
