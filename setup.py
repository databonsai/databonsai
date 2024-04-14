from setuptools import setup, find_packages

setup(
    name="databonsai",
    version="0.3.0",
    description="A package for cleaning and curating data with LLMs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Alvin Ryanputra",
    author_email="databonsai.ai@gmail.com",
    url="https://github.com/databonsai/databonsai",
    packages=find_packages(),
    install_requires=[
        "openai",
        "anthropic",
        "tenacity",
        "python-dotenv",
        "pydantic",
        "anthropic",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
