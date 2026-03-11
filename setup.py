from setuptools import setup, find_packages

setup(
    name="echo-feeling",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "requests"
    ],
    author="Phantom-rv7",
    description="Echo Feeling multimodal emotion analysis engine for ecommerce interaction",
)