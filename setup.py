from setuptools import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# import numpy


setup(
    name="notable2",
    version="2.0dev",
    author="Max Jacobi",
    author_email="mjacobi@theorie.ikp.physik.tu-darmstadt.de",
    packages=['notable2'],
    include_package_data=True,
    description="Some scripts to plot output from the ET",
    # cmdclass={'build_ext': build_ext},
    # ext_modules=[Extension("lic_internal", ["notable/lic_internal.pyx"], include_dirs=[numpy.get_include()])]
)
