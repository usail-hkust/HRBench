from setuptools import setup, find_packages

setup(
    name="hrbench",
    version="1.0.0",
    description="HRBench: Benchmarking Thinking-Mode Switch Strategies in Hybrid-Reasoning LLMs",
    author="HRBench Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "vllm>=0.4.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "jsonlines>=3.1.0",
    ],
)
