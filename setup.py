"""Setup instruction for the project's package."""

from setuptools import find_packages, setup

REQUIREMENTS = []

requirements_file = 'requirements.txt'

# Open and read the requirements.txt file
with open(requirements_file, 'r') as file:
    for line in file:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        REQUIREMENTS.append(line)

# DO NOT MODIFY
BASE_PACKAGES = [
    "typer",
    "pyyaml",
]

# DO NOT MODIFY
DEV_PACKAGES = [
    "pre-commit",
    "black",
    "isort",
    "flake8",
    "flake8-docstrings",
    "pep8-naming",
    "jupyterlab",
    "ipykernel",
]

# DO NOT MODIFY
TEST_PACKAGES = ["pytest", "pytest-cov"]

setup(
    name="fixmatch-on-places365",
    version="0.1.0",
    description="New project's short description",
    author="Zaid Ghazal",
    python_requires=">=3.9",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=BASE_PACKAGES + REQUIREMENTS,
    include_package_data=True,
    extras_require={"test": TEST_PACKAGES, "dev": TEST_PACKAGES + DEV_PACKAGES},
    # entry_points={"console_scripts": ["fixmatch-on-places365 = src.apps.cli:app"]},
)
