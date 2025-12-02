def test_exported_names_present():
    import txgraffiti2025.forms as f
    # Spot-check a bunch of public names are indeed exported
    for name in [
        "Expr","LinearForm","to_expr",
        "Predicate","GE","Between","Where",
        "Relation","Le","Ge","Eq","Conjecture",
        "Implication","Equivalence",
        "linear_le","product","with_floor","log_base","MonotoneRelation",
        "ClassInclusion","SubstructurePredicate",
    ]:
        assert hasattr(f, name), f"Missing export: {name}"
