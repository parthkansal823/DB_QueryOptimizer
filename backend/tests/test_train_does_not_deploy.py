"""
Training produces a candidate. Promoting one is a separate decision.

`retrain.py` states the invariant where it uses it -- *"train() writes to a
scratch path; promotion is this module's decision, not train()'s"* -- and
honours it by passing an explicit path. The CLI default did not: `MODEL_PATH`
was `models/plan_selector.pkl`, which is `model_store.CURRENT_PATH`, the file
`learned.py` loads to serve with.

So `python -m app.train` overwrote the served model with an untested one and the
registry's promotion decision was bypassed without a word. Measured: a retrain
whose offline eval was worse than the incumbent on every policy (risk-averse
headroom 0.045 against 0.365) was correctly *refused* promotion by the registry
and served anyway, because the pickle had already landed on the served path.
`model_store.py`'s docstring says the registry exists to stop exactly this --
"Until now the pipeline overwrote models/plan_selector.pkl on every train".

The check has to be an assertion rather than a convention, because the symptom
is silence: the registry keeps reporting the model it promoted, and a different
one answers every query.
"""

import os

import pytest

from app import model_store, train


class TestTheDefaultIsNotTheServedPath:
    def test_the_cli_default_does_not_point_at_the_served_model(self):
        assert os.path.abspath(train.MODEL_PATH) != os.path.abspath(model_store.CURRENT_PATH)

    def test_the_served_path_is_still_what_the_optimizer_loads(self):
        """
        Guards the other half: if `CURRENT_PATH` ever stopped being the file
        `learned.py` opens, the check above would pass while protecting nothing.
        """
        import inspect

        from app.optimizer.learned import LearnedOptimizer

        default = inspect.signature(LearnedOptimizer.__init__).parameters["model_path"].default
        assert os.path.abspath(default) == os.path.abspath(model_store.CURRENT_PATH)


class TestWritingToTheServedPathIsRefused:
    def test_an_explicit_served_path_raises(self, monkeypatch):
        """
        `--model-out models/plan_selector.pkl` is a caller asking `train()` to
        make a promotion decision it is not allowed to make. Refused loudly, not
        silently redirected: a run that believes it deployed a model and did not
        is the same silent failure pointing the other way.
        """
        with pytest.raises(ValueError) as excinfo:
            train.train(model_path=model_store.CURRENT_PATH, eval_path="models/_x.json")

        message = str(excinfo.value)
        assert "being served" in message
        assert "promote" in message.lower()

    def test_the_refusal_names_the_route_that_does_decide(self):
        """A refusal that does not say what to do instead just gets worked around."""
        with pytest.raises(ValueError) as excinfo:
            train.train(model_path=model_store.CURRENT_PATH, eval_path="models/_x.json")

        message = str(excinfo.value)
        assert "retrain" in message

    def test_a_scratch_path_is_not_refused(self, tmp_path, monkeypatch):
        """
        The refusal must be about the *served* path specifically. `retrain.py`
        calls `train()` on every retrain with a scratch path and has to keep
        working, so the guard must not be a blanket one.

        `_load_rows` is stubbed to fail immediately: the point is which error
        comes back, not to train a model.
        """
        monkeypatch.setattr(
            train, "_load_rows", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached the body"))
        )
        with pytest.raises(RuntimeError, match="reached the body"):
            train.train(
                model_path=str(tmp_path / "challenger.pkl"),
                eval_path=str(tmp_path / "e.json"),
            )
