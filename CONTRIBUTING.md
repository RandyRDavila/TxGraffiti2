# Contributing to GraffitiAI

Thank you for considering contributing to **GraffitiAI**! We welcome all types of contributions, from bug fixes and new features to documentation improvements and examples.

## Getting Started

1. **Fork the repository** on GitHub and clone your fork locally:

   ```bash
   git clone https://github.com/your-username/graffitiai.git
   cd graffitiai
   ```

2. **Create a virtual environment** and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate     # on Windows: .venv\\Scripts\\activate
   pip install -e .[dev]
   ```

3. **Verify the installation**:

   ```bash
   pytest --maxfail=1 --disable-warnings -q
   ```

## How to Contribute

### Reporting Bugs

* Search existing issues to see if the problem has already been reported.
* If not, open a new issue with:

  * A descriptive title.
  * Steps to reproduce.
  * Expected vs. actual behavior.
  * Environment information (`python --version`, OS).

### Suggesting Enhancements

* Use the issue tracker to propose new features or improvements.
* Clearly describe the use case and rationale.
* Provide examples or pseudocode if possible.

### Your First Code Contribution

1. **Create a new branch** from `main`:

   ```bash
   git checkout -b my-feature
   ```

2. **Make your changes** in the appropriate module under `graffitiai/`.

3. **Add or update tests** in the `tests/` directory.

4. **Update documentation** if your change affects the public API or usage:

   * README.md
   * Quickstart examples
   * Documentation site (if applicable)

5. **Run the full test suite**:

   ```bash
   pytest
   ```

6. **Commit and push**:

   ```bash
   git add .
   git commit -m "Add feature: description"
   git push origin my-feature
   ```

7. **Open a Pull Request** on GitHub against the `main` branch:

   * Follow the PR template.
   * Provide context and link to any relevant issues.
   * Request a review from at least one maintainer.

## Testing

* All new features should include unit tests.
* Place tests in the `txgraffiti/module_name/tests/` directory, mirroring the module structure.
* Use **pytest** fixtures for setup/teardown.
* Ensure code coverage remains high; aim for > 90% on new code.

## Documentation

* Keep docstrings up to date in modules and public classes/functions.
* Update `README.md` and `CONTRIBUTING.md` as needed.
* For larger changes, consider adding examples under `examples/`.

## Releasing

* Only maintainers can publish to PyPI.
* Follow semantic versioning (SEMVER).
* Tag the release on GitHub and push tags:

  ```bash
  git tag vX.Y.Z
  git push upstream --tags
  ```

---

Thanks for helping make GraffitiAI better! 🎉
