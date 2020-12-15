import pathlib
from setuptools import setup, find_packages

# The directory containing this file
ROOT = pathlib.Path(__file__).parent

# The text of the README file
README = (ROOT / "README.rst").read_text()

setup(
    name="pythonpredictions-cobra",
    version="1.0.1",
    description=("Package to build predictive models using linear"
                 "/logistic regression"),
    long_description=README,
    long_description_content_type="text/x-rst",
    packages=find_packages(include=['cobra', 'cobra.*']),
    url="https://github.com/PythonPredictions/cobra",
    license="MIT",
    author="Python Predictions",
    author_email="cobra@pythonpredictions.com",
    install_requires=[
        "numpy>=1.19.4",
        "pandas>=1.1.5",
        "scipy>=1.5.4",
        "scikit-learn>=0.23.1",
        "matplotlib>=3.3.3",
        "seaborn>=0.11.0"]
)
