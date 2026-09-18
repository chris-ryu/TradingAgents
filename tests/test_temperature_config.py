"""Tests for the configurable sampling temperature (#178/#168).

Temperature is a cross-provider knob: when set it must reach the underlying
chat client; when unset the provider keeps its own default.
"""

import importlib

import pytest

from tradingagents.llm_clients.factory import create_llm_client


@pytest.mark.unit
class TestTemperatureForwarding:
    @pytest.mark.parametrize(
        "provider,model",
        [
            # gpt-4.1 is intentionally a non-reasoning model: the GPT-5 family
            # are reasoning models and correctly drop temperature (see
            # test_openai_reasoning_effort), so forwarding is tested on gpt-4.1.
            ("openai", "gpt-4.1"),
            ("anthropic", "claude-sonnet-5"),
            ("google", "gemini-3.5-flash"),
            ("deepseek", "deepseek-chat"),
        ],
    )
    def test_temperature_reaches_client_when_set(self, provider, model):
        llm = create_llm_client(
            provider=provider, model=model, temperature=0.0, api_key="placeholder"
        ).get_llm()
        assert llm.temperature == 0.0

    def test_temperature_omitted_leaves_provider_default(self):
        # Not passing temperature must not force it to a value.
        llm = create_llm_client(
            provider="openai", model="gpt-4.1", api_key="placeholder"
        ).get_llm()
        # langchain's default is unset/None, not 0.0
        assert llm.temperature is None


@pytest.mark.unit
class TestTemperatureEnvOverlay:
    def test_env_sets_temperature(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.setenv("TRADINGAGENTS_TEMPERATURE", "0.2")
        importlib.reload(dc)
        # Stored on config (string from env is fine; consumed via float()).
        assert dc.DEFAULT_CONFIG["temperature"] in ("0.2", 0.2)
        assert float(dc.DEFAULT_CONFIG["temperature"]) == 0.2
        monkeypatch.delenv("TRADINGAGENTS_TEMPERATURE", raising=False)
        importlib.reload(dc)

    def test_default_temperature_is_none(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.delenv("TRADINGAGENTS_TEMPERATURE", raising=False)
        importlib.reload(dc)
        assert dc.DEFAULT_CONFIG["temperature"] is None


@pytest.mark.unit
class TestProviderKwargsTemperature:
    """_get_provider_kwargs float-coerces and forwards temperature, or omits it."""

    def _kwargs_for(self, temperature):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        # Call the method without constructing the full graph.
        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        graph.config = {"llm_provider": "openai", "temperature": temperature}
        return TradingAgentsGraph._get_provider_kwargs(graph)

    def test_float_string_coerced(self):
        assert self._kwargs_for("0.3")["temperature"] == 0.3

    def test_float_passthrough(self):
        assert self._kwargs_for(0.0)["temperature"] == 0.0

    def test_none_omitted(self):
        assert "temperature" not in self._kwargs_for(None)

    def test_empty_string_omitted(self):
        assert "temperature" not in self._kwargs_for("")


@pytest.mark.unit
class TestDeepTemperatureKwargs:
    """deep_temperature overrides temperature for the deep-think client only."""

    def test_deep_temperature_overrides_temperature(self):
        from tradingagents.graph.trading_graph import deep_llm_kwargs
        base = {"temperature": 0.3, "max_retries": 5, "callbacks": ["cb"]}
        deep = deep_llm_kwargs({"deep_temperature": 0.7}, base)
        assert deep == {"temperature": 0.7, "max_retries": 5, "callbacks": ["cb"]}
        assert base["temperature"] == 0.3  # the shared kwargs are not mutated

    def test_none_and_empty_inherit_temperature(self):
        from tradingagents.graph.trading_graph import deep_llm_kwargs
        base = {"temperature": 0.3}
        assert deep_llm_kwargs({"deep_temperature": None}, base) == {"temperature": 0.3}
        assert deep_llm_kwargs({"deep_temperature": ""}, base) == {"temperature": 0.3}
        assert deep_llm_kwargs({}, base) == {"temperature": 0.3}

    def test_env_string_is_coerced_to_float(self):
        from tradingagents.graph.trading_graph import deep_llm_kwargs
        assert deep_llm_kwargs({"deep_temperature": "0.7"}, {})["temperature"] == 0.7

    def test_deep_temperature_alone_sets_only_the_deep_side(self):
        # No shared temperature: the deep client gets one, the quick side keeps the provider default.
        from tradingagents.graph.trading_graph import deep_llm_kwargs
        assert deep_llm_kwargs({"deep_temperature": 0.7}, {}) == {"temperature": 0.7}


@pytest.mark.unit
class TestDeepTemperatureEnvOverlay:
    def test_env_sets_deep_temperature(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.setenv("TRADINGAGENTS_DEEP_TEMPERATURE", "0.7")
        importlib.reload(dc)
        assert float(dc.DEFAULT_CONFIG["deep_temperature"]) == 0.7
        monkeypatch.delenv("TRADINGAGENTS_DEEP_TEMPERATURE", raising=False)
        importlib.reload(dc)

    def test_default_deep_temperature_is_none(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.delenv("TRADINGAGENTS_DEEP_TEMPERATURE", raising=False)
        importlib.reload(dc)
        assert dc.DEFAULT_CONFIG["deep_temperature"] is None


@pytest.mark.unit
def test_graph_builds_deep_and_quick_clients_at_their_own_temperatures(monkeypatch, tmp_path):
    """The constructor must hand deep_temperature to the deep client and temperature to the quick one."""
    from unittest.mock import MagicMock

    import tradingagents.graph.trading_graph as tg

    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_llm(self):
            return MagicMock(name="llm")

    monkeypatch.setattr(tg, "create_llm_client", lambda **kw: FakeClient(**kw))
    # Keep the constructor off the network and away from graph compilation.
    monkeypatch.setattr(tg.TradingAgentsGraph, "_create_tool_nodes", lambda self: {})
    monkeypatch.setattr(tg, "GraphSetup", lambda *a, **k: MagicMock(name="setup"))
    monkeypatch.setattr(tg, "TradingMemoryLog", lambda cfg: MagicMock(name="memory"))

    config = dict(
        tg.DEFAULT_CONFIG,
        llm_provider="openai_compatible", backend_url="http://llm.test/v1",
        deep_think_llm="deep-x", quick_think_llm="quick-x",
        temperature=0.3, deep_temperature=0.7, checkpoint_enabled=False,
        results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
        memory_log_path=str(tmp_path / "memory.md"),
    )
    tg.TradingAgentsGraph(debug=False, config=config)

    by_model = {c["model"]: c for c in calls}
    assert by_model["deep-x"]["temperature"] == 0.7
    assert by_model["quick-x"]["temperature"] == 0.3
