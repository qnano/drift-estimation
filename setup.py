from setuptools import setup
from setuptools import Distribution

long_description= """
SMLM Drift estimation library, see publication:

Jelmer Cnossen, Tao Ju Cui, Chirlmin Joo, and Carlas Smith, 
"Drift correction in localization microscopy using entropy minimization," 
Opt. Express 29, 27961-27974 (2021)
https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-18-27961&id=457245
"""

# Tested with wheel v0.29.0
class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True

setup(
    name="dme",
    version="1.0",
    author="Jelmer Cnossen",
    author_email="j.cnossen@gmail.com",
    description="SMLM Drift Estimation using the DME algorithm",
    long_description=long_description,
 #   long_description_content_type="text/markdown",
#    url="https",
#    platforms=['nt'],
    classifiers=[
        "Programming Language :: Python :: 3",
		"Programming Language :: C++",
	    "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows"
    ],
    packages=['dme'],
    include_package_data=True,
    package_data={"dme": [
        "bin/release/dme_cpu.dll",
        "bin/release/dme_cuda.dll",
        "bin/release/msvcp140.dll",
        "bin/release/libdme_cuda.so"]},
    install_requires=[
		'numpy',
		'matplotlib',
		'tqdm',
        'scipy'
	]
    #distclass=BinaryDistribution
)

