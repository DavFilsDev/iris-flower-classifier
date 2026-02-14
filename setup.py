from setuptools import setup, find_packages

setup(
    name="iris-classifier",
    version="0.1.0",
    author="Fanampinirina Miharisoa David Fils RATIANDRAIBE",
    author_email="miharisoadavidfils@gmail.com",
    description="Iris flower classification using machine learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DavFilsDev/iris-flower-classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
)