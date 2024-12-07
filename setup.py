from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="birds_eye_view",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
)