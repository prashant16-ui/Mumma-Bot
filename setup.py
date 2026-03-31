from setuptools import find_packages, setup

setup(
    name="maa-saathi",
    version="0.1.0",
    author="Rishav Patel",
    author_email="krishav406@gmail.com@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-openai",
        "sentence-transformers",
        "pypdf"
    ],
)