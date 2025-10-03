"""
Setup script for EAI-RAIDS package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="eai-raids",
    version="2.0.0",
    author="EAI-RAIDS Team",
    author_email="contact@eai-raids.com",
    description="Enterprise Responsible AI Development System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hoangnhat2711/EAI-RAIDSs",
    project_urls={
        "Bug Tracker": "https://github.com/Hoangnhat2711/EAI-RAIDSs/issues",
        "Documentation": "https://eai-raids.readthedocs.io",
        "Source Code": "https://github.com/Hoangnhat2711/EAI-RAIDSs",
    },
    packages=find_packages(exclude=["tests", "benchmarks", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pylint>=2.12.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            "torch>=1.10.0",
            "tensorflow>=2.8.0",
            "psycopg2-binary>=2.9.0",
            "elasticsearch>=8.0.0",
            "boto3>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eai-raids=examples.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

