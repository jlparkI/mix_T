from setuptools import setup, find_packages, Extension
import os, numpy
from Cython.Distutils import build_ext

with open("README.md", "r") as fhandle:
    long_description = fhandle.read()

start_path = os.path.dirname(os.path.abspath(__file__))
mstep_funcs = Extension("optimized_mstep_functions",
                        sources=[os.path.join(start_path,
                        "studenttmixture", "optimized_mstep_functions.pyx")],
                        include_dirs = [numpy.get_include()])

setup(
        name="studenttmixture",
        version="0.0.2.6",
        packages=find_packages(),
        cmdclass = {"build_ext": build_ext},
        author="Jonathan Parkinson",
        author_email="jlparkinson1@gmail.com",
        description="Mixture modeling algorithms using the Student's t-distribution",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jlparki/mix_T",
        license="MIT",
        install_requires=["numpy>=1.19", "scipy>=1.6.0",
            "scikit-learn>=0.20.0"],
        ext_modules = [mstep_funcs]
)
