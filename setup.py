
exec(open("cobra/version.py").read())  # reads in __version__

import pathlib
from setuptools import setup, find_packages

# The directory containing this file
ROOT = pathlib.Path(__file__).parent

# The text of the README file
README = (ROOT / "README.rst").read_text()

setup(
    name="pythonpredictions-cobra",
    version=__version__,
    description=("A Python package to build predictive linear and logistic "
                 "regression models focused on performance and "
                 "interpretation."),
    long_description=README,
    long_description_content_type="text/x-rst",
    packages=find_packages(include=["cobra", "cobra.*"]),
    url="https://github.com/PythonPredictions/cobra",
    license="MIT",
    author="Python Predictions",
    author_email="cobra@pythonpredictions.com",
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scipy>=1.11.2",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.2",
        "tqdm>=4.62.2"]
)
