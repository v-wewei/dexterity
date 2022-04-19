import fnmatch
import os
import re
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

_here = Path(__file__).resolve().parent

name = "dexterity"

# Reference: https://github.com/patrick-kidger/equinox/blob/main/setup.py
with open(_here / name / "__init__.py") as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")


with open(_here / "README.md", "r") as f:
    readme = f.read()

core_requirements = [
    "absl-py",
    "numpy",
    "typing_extensions",
    "mujoco",
    "dm_control >= 1.0.1",
    "dm_robotics-geometry",
    "dm_robotics-transformations",
]

examples_requirements = [
    "matplotlib",
    "imageio",
    "imageio-ffmpeg",
]

testing_requirements = [
    "pytest-xdist",
]

dev_requirements = (
    [
        "black",
        "isort",
        "flake8",
        "mypy",
        "ipdb",
        "jupyter",
    ]
    + testing_requirements
    + examples_requirements
)

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

author = "Kevin Zakka"

author_email = "kevinarmandzakka@gmail.com"

description = "Software and tasks for dexterous multi-fingered hand manipulation, powered by MuJoCo"


# Reference: https://github.com/deepmind/dm_control/blob/main/setup.py
def find_data_files(package_dir, patterns, excludes=()):
    """Recursively finds files whose names match the given shell patterns."""
    paths = set()

    def is_excluded(s):
        for exclude in excludes:
            if fnmatch.fnmatch(s, exclude):
                return True
        return False

    for directory, _, filenames in os.walk(package_dir):
        if is_excluded(directory):
            continue
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                # NB: paths must be relative to the package directory.
                relative_dirpath = os.path.relpath(directory, package_dir)
                full_path = os.path.join(relative_dirpath, filename)
                if not is_excluded(full_path):
                    paths.add(full_path)

    return list(paths)


setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=f"https://github.com/kevinzakka/{name}",
    license="BSD",
    license_files=("LICENSE",),
    packages=find_packages(),
    package_data={
        "dexterity": find_data_files(
            package_dir="dexterity",
            patterns=["*.msh", "*.png", "*.skn", "*.stl", "*.xml", "*.typed"],
            excludes=[],
        ),
    },
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=core_requirements,
    classifiers=classifiers,
    extras_require={
        "testing": testing_requirements,
        "examples": examples_requirements,
        "dev": dev_requirements,
    },
)
