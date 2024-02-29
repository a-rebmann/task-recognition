from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="taskrecognition",
    packages=find_packages(),
    author='Adrian Rebmann',
    author_email='rebmann@uni-mannheim.de',
    version="0.1.25",
    description="long description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        "editdistance",
        "spacy",
        "pandas",
        "sklearn",
        "numpy",
        "pm4py==2.2.24",
        "jupyter",
        "nltk"
    ],
)