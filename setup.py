"""
Setup script for api_clients package.

Install with: pip install .
Or for development: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="api-clients",
    version="1.0.0",
    description="Unified API client framework for Scopus, Crossref, and other scholarly APIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/api-clients",
    packages=find_packages(),
    package_data={
        'api_clients': ['py.typed'],
    },
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.0.0",
        "requests>=2.25.0",
        "PyYAML>=5.4.0",
        "tqdm>=4.60.0",  # Progress bars (works in terminal and Jupyter)
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",  # For testing in notebooks
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
