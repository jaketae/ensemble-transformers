from setuptools import find_packages, setup


version = {}  # type: ignore

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

with open("ensemble_transformers/__version__.py", "r") as version_file:
    exec(version_file.read(), version)

with open("requirements.txt", "r") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name="ensemble-transformers",
    description="Ensembling Hugging Face Transformers made easy",
    version=version["version"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaketae/ensemble-transformers",
    author="Jake Tae",
    author_email="jaesungtae@gmail.com",
    install_requires=install_requires,
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)