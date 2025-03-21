from setuptools import setup, find_packages

setup(
    name="HomoTopiContinuation",
    version="1.0.0",
    author="Filippo Balzarin, Paolo Ginefra, Martina Missana",
    author_email="",
    description="A short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_package",
    packages=find_packages(where="src"),  # Specify the src directory
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
    ],
)
