"""FRED macro vendor: alias resolution, configuration errors, output formatting,
missing-value handling, lookahead-safe windowing, and router integration.

All API access is mocked, so these run without a network connection or a key.
"""
import copy
import unittest
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import fred, interface
from tradingagents.dataflows.config import set_config

# A small, stable set of observations to format against.
_META = {
    "seriess": [
        {
            "title": "Unemployment Rate",
            "units_short": "%",
            "frequency": "Monthly",
            "seasonal_adjustment_short": "SA",
        }
    ]
}
_OBS = {
    "observations": [
        {"date": "2025-06-01", "value": "4.1"},
        {"date": "2025-07-01", "value": "4.3"},
        {"date": "2025-08-01", "value": "."},   # missing -> skipped
        {"date": "2025-09-01", "value": "4.4"},
    ]
}


def _request_stub(meta=_META, obs=_OBS):
    """Build a _request replacement that dispatches on the endpoint path."""
    def _impl(path, params):
        if path == "series":
            return meta
        if path == "series/observations":
            return obs
        raise AssertionError(f"unexpected FRED path: {path}")
    return _impl


@pytest.mark.unit
class FredResolutionTests(unittest.TestCase):
    def test_alias_maps_to_series_id(self):
        self.assertEqual(fred._resolve_series_id("cpi"), "CPIAUCSL")
        self.assertEqual(fred._resolve_series_id("unemployment"), "UNRATE")

    def test_alias_is_case_and_separator_insensitive(self):
        self.assertEqual(fred._resolve_series_id("Fed Funds Rate"), "FEDFUNDS")
        self.assertEqual(fred._resolve_series_id("10y-treasury"), "DGS10")

    def test_unknown_alias_is_treated_as_raw_series_id(self):
        # Power users can pass any FRED series ID; we uppercase by convention.
        self.assertEqual(fred._resolve_series_id("dgs30"), "DGS30")
        self.assertEqual(fred._resolve_series_id("MyCustomSeries"), "MYCUSTOMSERIES")

    def test_descriptive_phrase_is_rejected(self):
        # An LLM phrase (spaces / too long) is not a series ID — reject up front
        # with guidance rather than 400ing the API.
        for bad in ("bank of japan rate", "the unemployment number", "X" * 31):
            with self.assertRaises(ValueError):
                fred._resolve_series_id(bad)

    def test_get_macro_data_returns_guidance_on_bad_indicator(self):
        # Invalid indicator -> actionable message, not a crash (no API call).
        out = fred.get_macro_data("bank of japan rate", "2026-01-01")
        self.assertIn("FRED", out)
        self.assertIn("not a known macro alias", out)


@pytest.mark.unit
class FredConfigTests(unittest.TestCase):
    def test_missing_key_raises_not_configured(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
                self.assertRaises(fred.FredNotConfiguredError):
            fred.get_api_key()

    def test_not_configured_is_a_value_error(self):
        # Routing relies on this subclassing for "vendor unavailable" handling.
        self.assertTrue(issubclass(fred.FredNotConfiguredError, ValueError))


@pytest.mark.unit
class FredFormattingTests(unittest.TestCase):
    def test_report_has_header_latest_change_and_table(self):
        with mock.patch.object(fred, "_request", side_effect=_request_stub()):
            out = fred.get_macro_data("unemployment", "2025-09-30", 365)
        self.assertIn("## FRED: Unemployment Rate (UNRATE)", out)
        self.assertIn("Units: %", out)
        self.assertIn("Frequency: Monthly (SA)", out)
        self.assertIn("**Latest:** 4.4 (2025-09-01)", out)
        # change over the window: 4.4 - 4.1 = +0.30
        self.assertIn("+0.30", out)
        self.assertIn("| 2025-06-01 | 4.1 |", out)

    def test_missing_value_is_skipped(self):
        with mock.patch.object(fred, "_request", side_effect=_request_stub()):
            out = fred.get_macro_data("unemployment", "2025-09-30", 365)
        # the "." observation must not appear as a row
        self.assertNotIn("2025-08-01", out)

    def test_empty_window_reports_no_observations(self):
        empty = {"observations": []}
        with mock.patch.object(fred, "_request", side_effect=_request_stub(obs=empty)):
            out = fred.get_macro_data("unemployment", "2025-09-30", 30)
        self.assertIn("No observations", out)

    def test_unknown_series_returns_not_found_message(self):
        # A well-formed but unknown series ID returns guidance, not a crash, so
        # the run is not aborted over an optional macro lookup.
        no_series = {"seriess": []}
        with mock.patch.object(fred, "_request", side_effect=_request_stub(meta=no_series)):
            out = fred.get_macro_data("ZZTOTALLYUNKNOWNX", "2025-09-30", 30)
        self.assertIn("not found", out)

    def test_long_series_is_truncated_but_change_uses_full_range(self):
        # Build > MAX_ROWS observations deterministically.
        obs = {
            "observations": [
                {"date": f"2025-01-{(i % 28) + 1:02d}", "value": str(i)}
                for i in range(fred.MAX_ROWS + 10)
            ]
        }
        with mock.patch.object(fred, "_request", side_effect=_request_stub(obs=obs)):
            out = fred.get_macro_data("unemployment", "2025-12-31", 365)
        self.assertIn(f"most recent {fred.MAX_ROWS}", out)
        # change-over-window must reference the true first (0) and last value
        self.assertIn("from 0 ", out)
        body_rows = [ln for ln in out.splitlines() if ln.startswith("| 2025")]
        self.assertEqual(len(body_rows), fred.MAX_ROWS)

    def test_window_is_lookahead_safe(self):
        # observation_end must equal curr_date so a past date never pulls future data.
        captured = {}

        def _capture(path, params):
            captured[path] = params
            return _META if path == "series" else _OBS

        with mock.patch.object(fred, "_request", side_effect=_capture):
            fred.get_macro_data("unemployment", "2025-09-30", 90)
        obs_params = captured["series/observations"]
        self.assertEqual(obs_params["observation_end"], "2025-09-30")
        self.assertEqual(obs_params["observation_start"], "2025-07-02")  # 90d back

    def test_requests_pin_the_data_vintage(self):
        # #1275: both the metadata and observations requests must pin the vintage
        # to curr_date (clamped to FRED's today), or FRED serves the latest
        # revision and revision-prone series leak future information. A past
        # curr_date sits below FRED's today, so it pins through unchanged.
        captured = {}

        def _capture(path, params):
            captured[path] = params
            return _META if path == "series" else _OBS

        with mock.patch.object(fred, "_fred_today", return_value="2026-01-01"), \
                mock.patch.object(fred, "_request", side_effect=_capture):
            fred.get_macro_data("cpi", "2025-09-30", 90)

        for path in ("series", "series/observations"):
            self.assertEqual(captured[path]["realtime_start"], "2025-09-30", path)
            self.assertEqual(captured[path]["realtime_end"], "2025-09-30", path)

    def test_future_curr_date_clamps_vintage_to_fred_today(self):
        # #1275 regression: on a live run curr_date is the caller's LOCAL date,
        # which can be a day ahead of FRED's US-Central clock. Pinning the vintage
        # to that future date 400s, and the routing layer then drops macro data
        # silently. The pin must clamp to FRED's today; the observation window
        # (future bars can't exist yet) stays at curr_date.
        captured = {}

        def _capture(path, params):
            captured[path] = params
            return _META if path == "series" else _OBS

        with mock.patch.object(fred, "_fred_today", return_value="2026-08-31"), \
                mock.patch.object(fred, "_request", side_effect=_capture):
            fred.get_macro_data("cpi", "2026-09-01", 90)  # local a day ahead of Chicago

        for path in ("series", "series/observations"):
            self.assertEqual(captured[path]["realtime_start"], "2026-08-31", path)
            self.assertEqual(captured[path]["realtime_end"], "2026-08-31", path)
        # the observation window still tracks curr_date, not the clamped vintage
        self.assertEqual(captured["series/observations"]["observation_end"], "2026-09-01")


@pytest.mark.unit
class FredRoutingTests(unittest.TestCase):
    def setUp(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def tearDown(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def test_macro_category_routes_to_fred(self):
        self.assertEqual(
            interface.get_category_for_method("get_macro_indicators"), "macro_data"
        )
        set_config({"data_vendors": {"macro_data": "fred"}})
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_indicators": {"fred": lambda *a, **k: "MACRO_OK"}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_macro_indicators", "cpi", "2026-06-01", 365)
        self.assertEqual(out, "MACRO_OK")

    def test_not_configured_degrades_gracefully(self):
        # macro_data is optional: with only fred and no key, the router degrades
        # to a sentinel instead of aborting the run — a missing optional key must
        # not crash an analysis.
        set_config({"data_vendors": {"macro_data": "fred"}})

        def _unconfigured(*a, **k):
            raise fred.FredNotConfiguredError("FRED_API_KEY not set")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_indicators": {"fred": _unconfigured}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_macro_indicators", "cpi", "2026-06-01", 365)
        self.assertIn("DATA_UNAVAILABLE", out)


@pytest.mark.unit
class FredSeriesIdHardeningTests(unittest.TestCase):
    """Pins the two FRED production defects seen in judge logs.

    Expected values are derived from FRED's documented series_id rule (the
    API itself answers 400 "Series IDs should be 25 or less alphanumeric
    characters") and from the observed 400 error_message "The series does
    not exist.", never from running this code.
    """

    def test_series_id_boundary_25_accepted_26_rejected(self):
        # FRED: series IDs are 25 or fewer alphanumeric characters. Both
        # sides of the boundary must hold locally so the guard rejects
        # exactly what the API rejects.
        self.assertEqual(fred._resolve_series_id("A" * 25), "A" * 25)
        with self.assertRaises(ValueError):
            fred._resolve_series_id("A" * 26)

    def test_raw_ids_with_punctuation_rejected_aliases_still_resolve(self):
        # Aliases are looked up BEFORE the plausibility guard and their keys
        # legitimately contain underscores/hyphens, so alias resolution must
        # keep working; the guard applies to RAW ids, where FRED requires
        # alphanumeric only.
        self.assertEqual(fred._resolve_series_id("10y_treasury"), "DGS10")
        self.assertEqual(fred._resolve_series_id("10y-treasury"), "DGS10")
        self.assertEqual(fred._resolve_series_id("fed_funds"), "FEDFUNDS")
        for bad in ("ABC_DEF", "ABC-DEF", "ABC,X", "ABC(X)"):
            with self.assertRaises(ValueError, msg=bad):
                fred._resolve_series_id(bad)

    def test_real_api_not_found_error_returns_guidance_not_raises(self):
        # The real FRED API answers an unknown series with HTTP 400 and
        # error_message "The series does not exist.", so _request raises
        # before any empty-seriess branch can run. get_macro_data must
        # convert that into the actionable not-found guidance it already
        # promises for the empty case, not let the ValueError abort the
        # optional macro lookup.
        def _raise_unknown(path, params):
            raise ValueError(
                "FRED request failed: Bad Request. The series does not exist."
            )

        # Control: the id itself resolves, so the first failure below can
        # only be the error-handling behaviour under test.
        self.assertEqual(fred._resolve_series_id("ZZTOPNOTREAL"), "ZZTOPNOTREAL")
        with mock.patch.object(fred, "_request", side_effect=_raise_unknown):
            out = fred.get_macro_data("ZZTOPNOTREAL", "2026-01-01", 30)
        self.assertIsInstance(out, str)
        self.assertIn("not found", out)
        self.assertIn("ZZTOPNOTREAL", out)

    def test_non_not_found_request_error_still_propagates(self):
        # Only the not-found 400 may be absorbed into guidance. Any other
        # _request failure (bad key, quota, transport) must keep raising so
        # the routing layer sees a vendor failure; a not-found-shaped
        # swallow here would mask real outages as "no such series".
        def _raise_auth(path, params):
            raise ValueError("FRED request failed: Bad Request. api_key is invalid")

        self.assertEqual(fred._resolve_series_id("ZZTOPNOTREAL"), "ZZTOPNOTREAL")
        with mock.patch.object(fred, "_request", side_effect=_raise_auth):
            with self.assertRaises(ValueError) as caught:
                fred.get_macro_data("ZZTOPNOTREAL", "2026-01-01", 30)
        self.assertIn("api_key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
