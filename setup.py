from setuptools import find_packages, setup

setup(
    name="shadow_hand",
    version="0.0.0",
    author="Kevin Zakka",
    license_files=("LICENSE",),
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "typing_extensions",
        "dm_control @ git+git://github.com/deepmind/dm_control.git",
        # TODO(kevin): Add dm_robotics-transformations and dm_robotics-geometry.
    ],
    extras_require={
        "testing": [
            "pytest",
            "absl-py",
        ],
    },
)
