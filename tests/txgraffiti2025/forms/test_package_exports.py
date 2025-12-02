def test_forms_star_import_smoke():
    # Importing the package should expose public API via __all__ without raising.
    from txgraffiti2025 import forms  # noqa: F401
