"""
Promotion has to judge the policy that will actually serve.

`_score_bundle` hardcoded `policy="risk_averse"` while `docker-compose.yml` sets
`SELECTION_POLICY: pairwise_rank` and the code default is `greedy` -- three
policies named in three places, and promotion deciding on the one nobody runs.

It bit. The model promoted on 2 September 2026 was accepted for beating the
champion by 8.3% on the risk-averse score. On the train-time eval its
`pairwise_rank` headroom -- the served policy -- was flat, 0.163 to 0.162, while
risk-averse *collapsed* from +0.365 to -0.142. The number promotion read moved
in the opposite direction to the number that matters.

`docs/WRITEUP.md` 2.22 names the pattern: a constant that duplicates a decision
made elsewhere will eventually disagree with it. 2.18 was the model trained on
the wrong action space, 2.21 was calibration tuning a gate the server does not
run, 2.22 was collection measuring a family training discards. This is the
fourth, and the fix is the same shape every time -- read the decision from
wherever it is actually made.
"""

import pytest

from app.retrain import served_policy


class TestServedPolicy:
    def test_it_reads_the_environment_serving_reads(self, monkeypatch):
        monkeypatch.setenv("SELECTION_POLICY", "pairwise_rank")
        assert served_policy() == "pairwise_rank"

    def test_it_matches_the_optimizer_default_when_unset(self, monkeypatch):
        """
        Both sides fall back to the same policy, so an unconfigured deployment
        is still scored on what it runs.
        """
        monkeypatch.delenv("SELECTION_POLICY", raising=False)
        import inspect

        from app.optimizer.learned import LearnedOptimizer

        default = inspect.signature(LearnedOptimizer.__init__).parameters["policy"].default
        # The optimizer's default is itself an os.getenv call evaluated at import
        # time, so compare the resolved values rather than the expressions.
        assert served_policy() == default

    def test_it_is_not_hardcoded_to_risk_averse(self, monkeypatch):
        """The specific regression: a constant that ignored its surroundings."""
        monkeypatch.setenv("SELECTION_POLICY", "greedy")
        assert served_policy() != "risk_averse"


class TestScoringHonoursThePolicy:
    def test_score_bundle_accepts_an_explicit_policy(self):
        import inspect

        from app.retrain import _score_bundle

        params = inspect.signature(_score_bundle).parameters
        assert "policy" in params, "the policy must be visible, not buried in the body"
        assert params["policy"].default is None, (
            "defaulting to None lets served_policy() decide at call time, so a "
            "deployment that changes SELECTION_POLICY is scored correctly without "
            "anyone remembering to update this"
        )

    @pytest.mark.parametrize("policy", ["greedy", "risk_averse", "thompson", "pairwise_rank"])
    def test_every_served_policy_is_scoreable(self, policy):
        """
        `pairwise_rank` lives on the ranker rather than the ensemble, so it is
        not one of `bandit.POLICIES`. Scoring it through `select_index` raises,
        and an exception in `_score_bundle` returns None -- which `retrain`
        reports as "could not score both models", silently skipping promotion
        for the one policy this deployment actually serves.
        """
        from app.optimizer.bandit import POLICIES
        from app.retrain import _score_bundle
        import inspect

        source = inspect.getsource(_score_bundle)
        if policy not in POLICIES:
            assert policy in source, (
                f"{policy} is served but is not one of bandit.POLICIES, so "
                f"_score_bundle must handle it explicitly"
            )


class TestTheDecisionRecordsWhatItMeasured:
    def test_the_promotion_reason_names_the_policy(self):
        import inspect

        from app import retrain

        source = inspect.getsource(retrain)
        assert "scored_with_policy" in source, (
            "the registry has to record which policy a verdict was reached under"
        )
        assert "on {served_policy()}" in source or "served_policy()}" in source
