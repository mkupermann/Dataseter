from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Dataseter - AI Training Dataset Creator"

# Core dependencies only - heavy/optional deps moved to extras_require
core_requirements = [
    "click>=8.1.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
    "pandas>=1.5.0",
    "numpy>=1.23.0",
    "fastapi>=0.95.0",
    "uvicorn>=0.21.0",
    "nltk>=3.8.0",
    "langdetect>=1.0.9",
    "ftfy>=6.1.0",
    "requests>=2.28.0",
    "aiohttp>=3.8.0",
    "tqdm>=4.65.0",
]

pdf_extras = [
    "PyPDF2>=3.0.0",
    "pdfplumber>=0.8.0",
    "pymupdf>=1.22.0",
    "ocrmypdf>=14.0.0",
    "pytesseract>=0.3.10",
    "Pillow>=9.4.0",
]

web_extras = [
    "scrapy>=2.8.0",
    "selenium>=4.8.0",
    "playwright>=1.30.0",
    "beautifulsoup4>=4.11.0",
]

office_extras = [
    "python-docx>=0.8.11",
    "openpyxl>=3.1.0",
    "python-pptx>=0.6.21",
    "ebooklib>=0.18",
    "mobi>=0.3.3",
]

ml_extras = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "sentence-transformers>=2.2.0",
    "spacy>=3.5.0",
]

dev_extras = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

all_extras = pdf_extras + web_extras + office_extras + ml_extras

setup(
    name="dataseter",
    version="1.0.0",
    author="Dataseter Team",
    author_email="dataseter@example.com",
    description="AI Training Dataset Creator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkupermann/dataseter",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
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
    install_requires=core_requirements,
    extras_require={
        "pdf": pdf_extras,
        "web": web_extras,
        "office": office_extras,
        "ml": ml_extras,
        "all": all_extras,
        "dev": dev_extras,
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
            "dataseter=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.html", "*.css", "*.js"],
    },
    zip_safe=False,
)
