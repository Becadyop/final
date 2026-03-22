from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="echo-feeling",
    version="2.0.0",
    author="Swathi Lekshmi SS, Romin Varghese, R Arun, Ivin Issac",
    description=(
        "Multimodal sentiment analysis for e-commerce: "
        "text + emoji + sticker fusion with ML classifiers."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Becadyop/final",
    packages=find_packages(exclude=["tests*", "training_phase*", "deployment_phase*"]),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "joblib>=1.3.0",
        "nltk>=3.8.0",
        "emoji>=2.10.0",
        "Pillow>=10.3.0",
    ],
    extras_require={
        "bert"  : ["transformers>=4.40.0", "torch>=2.2.0"],
        "server": ["flask>=3.0.0", "flask-cors>=4.0.0"],
        "dev"   : ["pytest>=8.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "sentiment-analysis", "nlp", "multimodal", "emoji",
        "ecommerce", "machine-learning", "bert",
    ],
    entry_points={
        "console_scripts": [
            "echo-feeling-train=training_phase.train:main",
            "echo-feeling-serve=deployment_phase.server:main",
        ],
    },
)
