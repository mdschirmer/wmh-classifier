from setuptools import setup

setup(
    name="wmh-classifier",
    version="1.0.0",
    py_modules=["wmh_classifier"],
    install_requires=["nibabel>=3.0.0", "numpy>=1.19.0", 
                     "scipy>=1.7.0", "scikit-image>=0.18.0"],
)
