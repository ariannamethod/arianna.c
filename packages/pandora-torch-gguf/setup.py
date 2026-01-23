"""
Pandora-Torch-GGUF — GGUF model vocabulary extraction

"Take the words from the tiny llama, leave the voice"
"""

from setuptools import setup, find_packages

setup(
    name="pandora-torch-gguf",
    version="0.1.0",
    description="GGUF vocabulary extraction for Arianna using TinyLlama 1.1B",
    author="ariannamethod",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "llama-cpp-python>=0.2.0",
        "huggingface_hub>=0.16.0",
    ],
    extras_require={
        "gpu": [
            # Install llama-cpp-python with CUDA support separately
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
