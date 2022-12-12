from setuptools import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# import numpy


setup(
    name="notable2",
    version="2.0dev",
    author="Maximilian Jacobi",
    author_email="mjacobi@theorie.ikp.physik.tu-darmstadt.de",
    packages=["notable2"],
    license='MIT',
    include_package_data=True,
    install_requires=[
        "numpy",
        "alpyne @ git+https://github.com/fguercilena/alpyne@master",
        "scipy",
        "matplotlib",
        "tqdm",
        "h5py",
    ],
    python_requires='>=3.7',
    description="Routines to post-process and plot output from the ET",
)
