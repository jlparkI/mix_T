from setuptools import setup, find_packages, Extension
import os, numpy

with open("README.md", "r") as fhandle:
    long_description = fhandle.read()

sqMahaExtension = Extension("squaredMahaDistance", sources=[os.path.join("studenttmixture",
                            "squared_maha_dist.c")], include_dirs=
                            [numpy.get_include()])

setup(
        name="studenttmixture",
        version="0.0.2.1",
        packages=find_packages(),
        author="Jonathan Parkinson",
        author_email="jlparkinson1@gmail.com",
        description="Mixture modeling algorithms using the Student's t-distribution",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jlparki/mix_T",
        license="MIT",
        install_requires=["numpy>=1.19", "scipy>=1.5.0"],
        ext_modules = [sqMahaExtension]
)
