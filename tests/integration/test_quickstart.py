from txgraffiti.playground    import ConjecturePlayground
from txgraffiti.generators    import ratios, convex_hull, linear_programming
from txgraffiti.heuristics    import dalmatian_accept, morgan_accept
from txgraffiti.processing    import remove_duplicates, sort_by_touch_count
from txgraffiti.example_data  import graph_data
from txgraffiti import Predicate

def test_quickstart_example_runs():
    ai = ConjecturePlayground(
        graph_data,
        object_symbol='G',
        base='connected',
    )

    regular = Predicate('regular', lambda df: df['min_degree'] == df['max_degree'])
    cubic = regular & (ai.max_degree == 3)
    small = ai.max_degree <= 3
    not_complete = ~ai.Kn

    ai.discover(
        methods         = [ratios, convex_hull, linear_programming],
        features        = [ai.independence_number],
        target          = ai.zero_forcing_number,
        hypothesis      = [cubic, small & not_complete],
        heuristics      = [dalmatian_accept, morgan_accept],
        post_processors = [remove_duplicates, sort_by_touch_count],
    )

    assert len(ai.conjectures) > 0
