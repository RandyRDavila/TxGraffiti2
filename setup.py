# setup.py
from setuptools import setup, find_packages

setup(
    name="txgraffiti2",
    version="0.1.0",
    description="Automated conjecture generation library",
    author="Randy Davila",
    author_email="rrd6@rice.edu",
    url="https://github.com/your-org/txgraffiti2",
    packages=find_packages(where="."),        # automatically finds your modules
    install_requires=[
        "pandas>=2.3.0",
        # list here any runtime dependencies you have
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
