from setuptools import setup, find_packages

with open("README.md", "r") as fhandle:
    long_description = fhandle.read()


setup(
        name="studenttmixture",
        version="0.0.1.1",
        packages=find_packages(),
        author="Jonathan Parkinson",
        author_email="jlparkinson1@gmail.com",
        description="Mixture modeling algorithms using the Student's t-distribution",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jlparki/mix_T",
        license="MIT",
        install_requires=["numpy>=1.20.2", "scipy>=1.6.2"]
)
