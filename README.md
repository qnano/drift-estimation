Example code for "Drift correction in localization microscopy using entropy minimization"
-----------------------------------------------------------------------------------------

Article link:
https://www.biorxiv.org/content/10.1101/2021.03.30.437682v1

There are two options:

* Standalone example, able to run on CUDA-free PCs.
  See dme_example.py
  
* Example using photonpy library, which demonstrates the full SMLM pipeline. Currently requires CUDA, but we're working on making that optional:
  See dme_photonpy_example.py
 

To run dme_example.py:
----------------------

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

To run dme_photonpy_example.py
------------------------------

1. Install CUDA toolkit 11.2
2. Install photonpy library. Only windows 64-bit at the moment:

```
pip install photonpy=1.0.39
```
