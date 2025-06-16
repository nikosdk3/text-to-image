from setuptools import find_packages, setup

setup(
    name="text_to_image_pytorch",
    packages=find_packages(exclude=[]),
    version="0.0.2",
    license="MIT",
    description="Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding",
    author="Nikos Kotsopulos",
    author_email="nikomvp03@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/nikosdk3/text-to-image",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "diffusion",
        "machine learning",
        "text-to-image",
    ],
    requires=["torch" "transformers"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
)
