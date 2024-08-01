from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ragsynth",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        requirements
    ],
    author="Bassim Eledath",
    author_email="bassimfaizal@gmail.com",
    description="Quickly generate synthetic evaluation datasets for RAG projects",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bassimeledath/RAGSynth",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
