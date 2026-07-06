# FILE: setup.py (18 lines)
from setuptools import setup, find_packages

setup(
    name="portfolio-optimization",
    version="0.1.0",
    description="JEPA-based portfolio optimization with MPC",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pandas",
        "pyarrow",
    ],
)
