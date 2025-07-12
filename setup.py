# # setup.py
# from setuptools import setup, find_packages

# setup(
#     name="txgraffiti",
#     version="0.1.0",
#     description="Automated conjecture generation library",
#     author="Randy Davila",
#     author_email="rrd6@rice.edu",
#     url="https://github.com/your-org/txgraffiti2",
#     packages=find_packages(where="."),        # automatically finds your modules
#     install_requires=[
#         "pandas>=2.3.0",
#         "pytest",
#         "scipy",
#         "pulp",
#         # list here any runtime dependencies you have
#     ],
#     python_requires=">=3.8",
#     include_package_data=True,
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "Operating System :: OS Independent",
#     ],
#     package_data={
#         # this says “include any .csv under example_data”
#         "graffitiai.example_data": ["*.csv"],
#     },
# )
from setuptools import setup, find_packages

setup(
    name="txgraffiti",
    version="0.1.3",
    author="Randy Davila",
    description="Automated conjecturing system for mathematics and graph theory.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RandyRDavila/txgraffiti2",
    packages=find_packages(),
    # install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
