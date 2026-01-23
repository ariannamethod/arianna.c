"""
Pandora-Torch — PyTorch vocabulary extraction with LoRA delta support

"Take the words, leave the voice"
"""

from setuptools import setup, find_packages

setup(
    name="pandora-torch",
    version="0.1.0",
    description="PyTorch vocabulary extraction for Arianna",
    author="ariannamethod",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
    ],
    extras_require={
        "full": [
            "transformers>=4.30",
        ],
        "stanley": [
            # Stanley is installed from git
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
