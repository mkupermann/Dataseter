from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dataseter",
    version="1.0.0",
    author="Dataseter Team",
    author_email="dataseter@example.com",
    description="Advanced AI Training Dataset Creator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dataseter",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
        ],
        "distributed": [
            "ray>=2.0.0",
            "dask[complete]>=2023.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dataseter=cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.html", "*.css", "*.js"],
    },
    zip_safe=False,
)