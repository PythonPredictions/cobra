from distutils.core import setup


setup(
    name="cobra",
    version="1.0.0",
    description="Python Prediction's methodology for predictive analytics",
    packages=["cobra", "cobra.preprocessing", "cobra.model_building"],
    url="https://github.com/PythonPredictions",
    #long_description=long_description,  # TO DO
    #long_description_content_type="text/markdown",
    install_requires=[
        "pandas>=0.25.1",
        "numpy>=1.17.2",
        "scipy>=1.2.0",
        "scikit_learn>=0.22.1",
        "matplotlib>=3.0.2",
        "seaborn>=0.9.0"],
    python_requires=">=3.5",
)
