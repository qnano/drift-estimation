from setuptools import setup

long_description = """
SMLM drift estimation by entropy minimization (DME), GPU implementation in PyTorch.

Jelmer Cnossen, Tao Ju Cui, Chirlmin Joo, and Carlas Smith,
"Drift correction in localization microscopy using entropy minimization,"
Opt. Express 29, 27961-27974 (2021)
https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-18-27961&id=457245
"""

setup(
    name="dme",
    version="2.0.0",
    author="Jelmer Cnossen",
    author_email="j.cnossen@gmail.com",
    description="SMLM drift estimation using the DME algorithm",
    long_description=long_description,
    url="https://github.com/qnano/drift-estimation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=['dme'],
    include_package_data=True,
    package_data={"dme": ["weights/temporal_prior.pt"]},   # learned temporal prior of method='v2' 
    python_requires='>=3.8',
    install_requires=['numpy', 'scipy', 'matplotlib', 'torch'],
    extras_require={
        # fused pair kernels for method='v1' (pick the cupy build matching your CUDA driver)
        'cupy': ['cupy-cuda12x'],
    },
)
