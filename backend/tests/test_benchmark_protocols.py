

# -- the summary leads with the figures nobody selected for --------------


def test_the_unbiased_lines_come_before_the_headline(capsys):
    """
    `headroom captured` is over all 59 workload queries, 27 of which were
    generated and then *kept* because the best plan in the served action space
    beat native by at least 8%. That was the right rule for training coverage
    and it makes the number partly a measurement of the selection.

    Leading with it is how a project quotes its most flattering figure without
    meaning to. Held-out and curated answer "does this work"; all-queries
    answers "what does this workload look like".
    """
    from app.benchmark import _summarise

    _summarise(
        [
            {
                "captured_pct": 41.3,
                "held_out_captured_pct": 45.7,
                "n_held_out": 15,
                "curated_captured_pct": 33.0,
                "n_curated": 32,
            }
        ],
        "compare",
        1,
    )
    out = capsys.readouterr().out

    held = out.index("held-out (never trained on)")
    curated = out.index("curated (not selected)")
    everything = out.index("all queries")

    assert held < everything, "the biased figure is being led with"
    assert curated < everything, "the biased figure is being led with"


def test_the_selection_effect_is_named_not_left_to_arithmetic(capsys):
    """A reader should not have to subtract two lines to find out that one of
    them is measured on a benchmark chosen for being winnable."""
    from app.benchmark import _summarise

    _summarise(
        [
            {
                "captured_pct": 41.3,
                "held_out_captured_pct": 45.7,
                "n_held_out": 15,
                "curated_captured_pct": 33.0,
                "n_curated": 32,
            },
            {
                "captured_pct": 43.0,
                "held_out_captured_pct": 50.4,
                "n_held_out": 15,
                "curated_captured_pct": 34.2,
                "n_curated": 32,
            },
        ],
        "compare",
        2,
    )
    out = capsys.readouterr().out

    assert "selection effect" in out
    assert "+8.5 point gap" in out or "+8.6 point gap" in out


def test_no_selection_note_when_there_is_nothing_to_compare(capsys):
    """A workload with no generated queries has no selection effect to report,
    and inventing a zero-point gap would imply one was measured."""
    from app.benchmark import _summarise

    _summarise([{"captured_pct": 41.3, "n_held_out": 0}], "compare", 1)
    out = capsys.readouterr().out

    assert "selection effect" not in out
