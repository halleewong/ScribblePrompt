from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="scribbleprompt",
    version="0.1.0",
    python_requires=">=3.9",
    packages=find_packages(exclude="notebooks"),
    install_requires=requirements,
)
