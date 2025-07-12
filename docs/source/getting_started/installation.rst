Installation
=====================

TxGraffiti supports Python 3.8 and above, and works on Linux, macOS, and Windows.

Prerequisites
-------------

- **Python 3.8+**
- **pip** (preferably the latest version: `pip install --upgrade pip`)
- Optional system‐wide solvers for LP generators:
  - **CBC** (`brew install cbc` or `apt-get install coinor-cbc`)
  - **GLPK** (`brew install glpk` or `apt-get install glpk`)

Quick install from PyPI
-----------------------

Install the latest stable release:

.. code-block:: bash

   pip install txgraffiti

If you plan to use LP‐based generators, make sure you have installed CBC or GLPK on your system (see above).

Installing with Extras
----------------------

To pull in all optional dependencies (e.g. for LP, convex‐hull, exporting to Lean):

.. code-block:: bash

   pip install "txgraffiti[all]"

This will also install:

- `pulp` (LP interface)
- `scipy` (convex hull)
- `networkx` (example data)
- `lean4-export` (Lean stub exporter)

Development Installation
------------------------

To work on the source or run the tests:

.. code-block:: bash

   git clone https://github.com/RandyRDavila/TxGraffiti2.git
   cd txgraffiti
   # create an isolated environment (recommended)
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate.bat     # Windows
   pip install --upgrade pip
   pip install -e ".[dev]"        # installs txgraffiti plus dev dependencies

Dev extras include:
- `pytest` & `pytest-cov` for running tests
- `black` & `flake8` for linting
- `sphinx` for building docs



Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/RandyRDavila/TxGraffiti2.git
   cd txgraffiti
   pip install -e .

Once installed, verify your setup:

.. code-block:: bash

   python - <<EOF
   import txgraffiti
   print("TxGraffiti version:", txgraffiti.__version__)
   EOF

You’re now ready to start discovering conjectures!
