from setuptools import setup, find_packages

setup(
    name="tervid",
    version="0.1.0",
    description="TerViD: 1.58-bit Ternary Video Diffusion Transformers",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "diffusers>=0.28.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "einops>=0.7.0",
    ],
)
