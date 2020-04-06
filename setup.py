from distutils.core import setup


setup(
    name="cobra",
    version="1.0.0",
    description="Python Prediction's methodology for predictive analytics",
    packages=["cobra",
              "cobra.preprocessing",
              "cobra.model_building",
              "cobra.evaluation"],
    url="https://github.com/PythonPredictions",
    install_requires=[
        "numpy>=1.17.2",
        "pandas>=0.25.1",
        "scipy>=1.2.0",
        "scikit_learn>=0.22.1",
        "matplotlib>=3.0.2",
        "seaborn>=0.9.0"],
    python_requires=">=3.6",
)
