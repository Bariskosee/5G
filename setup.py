"""Setup configuration for TEKNOFEST 2026 — 5G & AI Smart Road Safety."""

from setuptools import setup, find_packages

setup(
    name="smart-road-safety",
    version="0.1.0",
    description="TEKNOFEST 2026 — 5G & AI Smart Road Safety (Phase 1)",
    author="Team Placeholder",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "ultralytics",
        "opencv-python",
        "easyocr",
        "pytesseract",
        "albumentations",
        "Pillow",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pyyaml",
        "tqdm",
        "decord",
    ],
    extras_require={
        "dev": ["pytest", "jupyterlab"],
    },
)
