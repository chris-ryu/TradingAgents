"""FRED (Federal Reserve Economic Data) macro vendor.

Fetches macroeconomic time series — policy rates, Treasury yields, inflation,
labor, growth — from the St. Louis Fed's free API. Used by the news analyst to
ground macro commentary in actual numbers rather than headlines alone.

A free API key (https://fred.stlouisfed.org/docs/api/api_key.html) is read from
``FRED_API_KEY``; if it is unset the vendor raises ``FredNotConfiguredError`` so
the routing layer treats it as "unavailable" rather than a hard crash.
"""
import logging
import os
import re
from datetime import datetime, timedelta

import pytz
import requests

from .errors import VendorNotConfiguredError

logger = logging.getLogger(__name__)

FRED_API_BASE = "https://api.stlouisfed.org/fred"

# FRED's realtime clock runs on US Central (St. Louis Fed). It rejects a
# realtime date in its own future with a 400, so the vintage pin is clamped to
# this rather than the caller's local date (#1275). pytz (already a dependency)
# bundles its own tz database, so this works where system tzdata is absent.
FRED_TZ = pytz.timezone("America/Chicago")

# Network timeout (seconds) so a stalled request can't hang the agents,
# mirroring the Alpha Vantage client.
REQUEST_TIMEOUT = 30

# Default trailing window when the caller does not specify one. A year captures
# the trend and the year-over-year base for most monthly/quarterly series.
DEFAULT_LOOKBACK_DAYS = 365

# Rows cap for the rendered table: recent values matter most for a decision, and
# daily series (yields, VIX) over a long window would otherwise flood context.
MAX_ROWS = 40

# FRED's own series_id rule, mirrored locally: 25 or fewer alphanumeric
# characters. The API answers 400 "Series IDs should be 25 or less alphanumeric
# characters", so the guard below rejects exactly that set instead of paying a
# round trip for it. ASCII-only on purpose: str.isalnum() also accepts non-ASCII
# digits and letters (for example "2" or roman numerals) that FRED does not.
SERIES_ID_PATTERN = re.compile(r"[A-Z0-9]{1,25}")

# _request renders every FRED HTTP 400 as ValueError("<prefix><error_message>"),
# which is the only signal a caller can classify on. One constant, so _request
# and the classifier below cannot drift apart.
REQUEST_ERROR_PREFIX = "FRED request failed: "

# FRED's documented wording for an unknown series, and the only wording treated
# as "not found". Key/auth/quota/parameter failures read "api_key is invalid",
# "must be a date" and the like; none of them claims non-existence.
NOT_FOUND_MESSAGE_MARKERS = ("the series does not exist",)

# Curated human-friendly aliases -> FRED series IDs. Anything not listed is used
# verbatim as a raw FRED series ID, so power users are never limited to this set.
MACRO_SERIES = {
    # Policy rate & Treasury yields
    "fed_funds_rate": "FEDFUNDS",
    "federal_funds_rate": "FEDFUNDS",
    "fed_funds": "FEDFUNDS",
    "2y_treasury": "DGS2",
    "10y_treasury": "DGS10",
    "30y_treasury": "DGS30",
    "10y_2y_spread": "T10Y2Y",
    "yield_curve": "T10Y2Y",
    # Inflation
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "pce": "PCEPI",
    "core_pce": "PCEPILFE",
    "inflation_expectations": "T10YIE",
    # Growth & output
    "real_gdp": "GDPC1",
    "gdp": "GDP",
    "industrial_production": "INDPRO",
    # Labor
    "unemployment_rate": "UNRATE",
    "unemployment": "UNRATE",
    "nonfarm_payrolls": "PAYEMS",
    "payrolls": "PAYEMS",
    "initial_claims": "ICSA",
    # Money & markets
    "m2": "M2SL",
    "money_supply": "M2SL",
    "vix": "VIXCLS",
    "dollar_index": "DTWEXBGS",
    # Sentiment & housing
    "consumer_sentiment": "UMCSENT",
    "housing_starts": "HOUST",
    "retail_sales": "RSAFS",
}


class FredNotConfiguredError(VendorNotConfiguredError):
    """Raised when FRED is selected but no API key is configured.

    A VendorNotConfiguredError (and thus still a ValueError), so the routing
    layer's "vendor unavailable" handling and existing ValueError callers both
    keep working.
    """


def get_api_key() -> str:
    """Retrieve the FRED API key from the environment."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise FredNotConfiguredError(
            "FRED_API_KEY environment variable is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html."
        )
    return api_key


def _resolve_series_id(indicator: str) -> str:
    """Map a friendly alias to a FRED series ID, or pass a raw ID through.

    Raises ``ValueError`` when the input is neither a known alias nor a plausible
    series ID — typically a descriptive phrase the LLM passed instead (e.g.
    "bank of japan rate"). FRED IDs are short and alphanumeric, so this rejects
    it up front with guidance rather than letting it 400 the API.
    """
    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")
    if key in MACRO_SERIES:
        return MACRO_SERIES[key]
    candidate = indicator.strip().upper()
    # FRED series IDs are 25 or fewer alphanumeric characters; reject anything
    # else — a descriptive phrase the LLM passed, or an id carrying punctuation
    # the API would 400 — rather than paying for the failure. The alias lookup
    # above is what lets keys like "10y_treasury" resolve, so the underscore and
    # hyphen belong to the alias namespace only, never to a raw id.
    if SERIES_ID_PATTERN.fullmatch(candidate) is None:
        raise ValueError(
            f"'{indicator}' is not a known macro alias or a valid FRED series ID. "
            f"Use an alias (e.g. 'cpi', 'unemployment', '10y_treasury') or a raw "
            f"FRED series ID (e.g. 'CPIAUCSL')."
        )
    return candidate


def _fred_today() -> str:
    """FRED's current calendar date (US Central) as ``yyyy-mm-dd``.

    The vintage pin is clamped to this: FRED rejects a ``realtime_start`` after
    its own today with a 400, and ``curr_date`` on a live run comes from the
    caller's local clock, which can already be tomorrow in Chicago.
    """
    return datetime.now(FRED_TZ).strftime("%Y-%m-%d")


def _request(path: str, params: dict) -> dict:
    """GET a FRED endpoint, surfacing FRED's JSON error body on a bad request."""
    api_params = {**params, "api_key": get_api_key(), "file_type": "json"}
    response = requests.get(
        f"{FRED_API_BASE}/{path}", params=api_params, timeout=REQUEST_TIMEOUT
    )
    # FRED returns 400 with a JSON {"error_message": ...} for unknown series IDs
    # or malformed params; turn that into a clear, actionable error.
    if response.status_code == 400:
        try:
            message = response.json().get("error_message", response.text)
        except ValueError:
            message = response.text
        raise ValueError(f"{REQUEST_ERROR_PREFIX}{message}")
    response.raise_for_status()
    return response.json()



def _not_found_guidance(series_id: str) -> str:
    """Actionable message for a series FRED does not have.

    Shared by both ways that happens: the HTTP 400 path and an HTTP 200 whose
    ``seriess`` list came back empty.
    """
    return (
        f"FRED series '{series_id}' not found. Pass a known alias "
        f"(e.g. 'cpi', 'unemployment') or a valid FRED series ID."
    )


def _is_unknown_series_error(exc: ValueError) -> bool:
    """True only for FRED's own "this series does not exist" 400.

    _request flattens every 400 into a ValueError carrying FRED's error_message,
    so text is the only discriminator here. Two conditions must both hold, and
    they are deliberately narrow:

      * the text starts with the prefix _request itself adds, which an error
        raised anywhere else cannot carry — a missing API key
        (FredNotConfiguredError, also a ValueError), a transport failure or some
        other module's ValueError is excluded before the wording is even read;
      * the remainder contains FRED's documented not-found wording, a statement
        about the series identity. An invalid or missing key, a quota refusal or
        a malformed parameter never says a series "does not exist".

    So an auth or network outage cannot be mistaken for "no such series"; it
    keeps raising, and the routing layer still reports a vendor failure rather
    than losing macro data silently.
    """
    if isinstance(exc, FredNotConfiguredError):
        return False
    text = str(exc)
    if not text.startswith(REQUEST_ERROR_PREFIX):
        return False
    detail = text[len(REQUEST_ERROR_PREFIX) :].lower()
    return any(marker in detail for marker in NOT_FOUND_MESSAGE_MARKERS)


def get_macro_data(
    indicator: str,
    curr_date: str,
    look_back_days: int | None = None,
) -> str:
    """Fetch a FRED macroeconomic series as a formatted markdown report.

    Args:
        indicator: A friendly alias (e.g. "cpi", "unemployment", "10y_treasury")
            or a raw FRED series ID (e.g. "CPIAUCSL", "DGS10").
        curr_date: The as-of date (yyyy-mm-dd). It bounds the observation window
            AND pins the data vintage: FRED is queried with the realtime bounds
            set to ``curr_date`` (clamped to FRED's own today) so a historical
            run sees the values that were actually published by that date, not
            later revisions. Without this, revision-prone series (CPI, GDP, ...)
            would leak future information into a backtest (#1275).
        look_back_days: Trailing window length; ``None`` uses DEFAULT_LOOKBACK_DAYS.

    Returns:
        A markdown report with the series title, units, frequency, the latest
        value, the change over the window, and a recent observation table.
    """
    if look_back_days is None:
        look_back_days = DEFAULT_LOOKBACK_DAYS

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    # Pin the data vintage. FRED defaults both realtime bounds to today, serving
    # the LATEST revision of every observation; a single-day realtime interval
    # asks for the values known as of the pin instead, on both the metadata and
    # observations requests (#1275). Clamp to FRED's today: on a live run
    # curr_date is the caller's local date, which can be a day ahead of Chicago,
    # and a realtime date in FRED's future 400s -> the routing layer would then
    # drop macro data silently. A past curr_date is unaffected, so historical
    # point-in-time behaviour is preserved.
    pit = min(curr_date, _fred_today())
    realtime = {"realtime_start": pit, "realtime_end": pit}

    # Invalid LLM-supplied indicator: return guidance rather than raising, so a
    # bad argument doesn't abort the run (the routing layer also degrades macro
    # data, but a specific message is more useful to the analyst).
    try:
        series_id = _resolve_series_id(indicator)
    except ValueError as e:
        return f"FRED: {e}"

    # The real FRED API answers an unknown series with HTTP 400 ("The series
    # does not exist."), so _request raises before any empty-seriess branch can
    # run. Only that specific error degrades into the guidance the empty case
    # already returns; any other failure keeps raising so the routing layer sees
    # a vendor failure instead of silently dropping macro data.
    try:
        payload = _request("series", {"series_id": series_id, **realtime})
    except ValueError as exc:
        if not _is_unknown_series_error(exc):
            raise
        return _not_found_guidance(series_id)
    meta = payload.get("seriess") or []
    if not meta:
        return _not_found_guidance(series_id)
    info = meta[0]
    title = info.get("title", series_id)
    units = info.get("units_short") or info.get("units", "")
    frequency = info.get("frequency", "")
    seasonal = info.get("seasonal_adjustment_short", "")

    observations = _request(
        "series/observations",
        {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": curr_date,
            "sort_order": "asc",
            **realtime,
        },
    ).get("observations", [])

    # FRED encodes a missing observation as ".".
    points = [
        (o["date"], o["value"])
        for o in observations
        if o.get("value") not in (".", None, "")
    ]

    header = (
        f"## FRED: {title} ({series_id})\n"
        f"- Units: {units}\n"
        f"- Frequency: {frequency}"
        f"{f' ({seasonal})' if seasonal else ''}\n"
        f"- Window: {start_date} to {curr_date}\n"
    )

    if not points:
        return header + (
            f"\nNo observations for {series_id} in this window at the {pit} "
            f"vintage. The series may report less frequently than the window "
            f"(try a longer look_back_days), or have no vintage published by "
            f"then (unpublished as of {pit}, or before ALFRED coverage begins)."
        )

    first_date, first_val = points[0]
    last_date, last_val = points[-1]
    try:
        delta = float(last_val) - float(first_val)
        base = float(first_val)
        pct = f" ({delta / base * 100:+.2f}%)" if base != 0 else ""
        summary = (
            f"\n**Latest:** {last_val} ({last_date}) | "
            f"**Change over window:** {delta:+.2f}{pct} "
            f"from {first_val} ({first_date})\n"
        )
    except ValueError:
        summary = f"\n**Latest:** {last_val} ({last_date})\n"

    shown = points
    note = ""
    if len(points) > MAX_ROWS:
        shown = points[-MAX_ROWS:]
        note = f"\n_(showing the most recent {MAX_ROWS} of {len(points)} observations)_\n"

    table = (
        "\n| Date | Value |\n| --- | --- |\n"
        + "\n".join(f"| {d} | {v} |" for d, v in shown)
        + "\n"
    )

    return header + summary + note + table
