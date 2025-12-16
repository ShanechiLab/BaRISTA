from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="barista",
    version="1.0.0",
    description="PyTorch implementation of BaRISTA: Brain Scale Informed Spatiotemporal Representation of Human Intracranial Neural Activity",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Lucine L. Oganesian, Saba Hashemi, Maryam M. Shanechi",
    author_email="shanechi@usc.edu",
    url="https://github.com/ShanechiLab/BaRISTA",  # change to actual repo URL
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "barista-train=barista.train:main",
            "barista-prepare=barista.prepare_segments:main",
        ],
    },
)
