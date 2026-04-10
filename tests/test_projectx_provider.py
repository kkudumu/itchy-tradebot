from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.config.models import InstrumentsConfig, ProjectXConfig
from src.providers.projectx import (
    ProjectXClient,
    ProjectXHistoricalDataLoader,
    ProjectXMarketDataProvider,
)


class _MockResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _MockSession:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def post(self, url: str, json: dict, headers: dict, timeout: float) -> _MockResponse:
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        payload = self.responses.pop(0)
        return _MockResponse(payload.get("status_code", 200), payload["json"])


def test_projectx_client_login_and_bearer_header():
    session = _MockSession(
        [
            {"json": {"token": "abc", "success": True, "errorCode": 0, "errorMessage": None}},
            {"json": {"accounts": [], "success": True, "errorCode": 0, "errorMessage": None}},
        ]
    )
    client = ProjectXClient(
        base_url="https://api.thefuturesdesk.projectx.com",
        username="user",
        api_key="key",
        session=session,
    )

    client.search_accounts()

    assert session.calls[0]["url"].endswith("/api/Auth/loginKey")
    assert session.calls[1]["headers"]["Authorization"] == "Bearer abc"


def test_projectx_client_refreshes_expiring_token():
    session = _MockSession(
        [
            {"json": {"newToken": "refreshed", "success": True, "errorCode": 0, "errorMessage": None}},
            {"json": {"accounts": [], "success": True, "errorCode": 0, "errorMessage": None}},
        ]
    )
    client = ProjectXClient(
        base_url="https://api.thefuturesdesk.projectx.com",
        username="user",
        api_key="key",
        session=session,
        token_refresh_buffer_seconds=300,
    )
    client._token = "stale"
    client._token_expiry = datetime.now(timezone.utc)

    client.search_accounts()

    assert session.calls[0]["url"].endswith("/api/Auth/validate")
    assert session.calls[1]["headers"]["Authorization"] == "Bearer refreshed"


def test_projectx_market_provider_normalizes_bars():
    class StubClient:
        def retrieve_bars(self, **kwargs):
            return {
                "bars": [
                    {"t": "2026-01-01T00:01:00Z", "o": 2, "h": 3, "l": 1, "c": 2.5, "v": 20},
                    {"t": "2026-01-01T00:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
                ],
                "success": True,
                "errorCode": 0,
                "errorMessage": None,
            }

        def search_contract_by_id(self, contract_id: str):
            return {
                "contract": {
                    "id": contract_id,
                    "symbolId": "F.US.TEST",
                    "name": "TEST",
                    "description": "Test contract",
                    "tickSize": 0.25,
                    "tickValue": 5.0,
                },
                "success": True,
                "errorCode": 0,
                "errorMessage": None,
            }

    instruments = InstrumentsConfig.model_validate(
        {"instruments": [{"symbol": "XAUUSD", "contract_id": "CON.TEST", "default_quantity": 1}]}
    )
    provider = ProjectXMarketDataProvider(
        client=StubClient(),
        config=ProjectXConfig(),
        instruments=instruments,
    )

    df = provider.fetch_bars("XAUUSD", "1M", limit=2)

    assert list(df.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert list(df["close"]) == [1.5, 2.5]
    spec = provider.get_contract_spec("XAUUSD")
    assert spec.point_value == 20.0


def test_projectx_historical_loader_returns_indexed_dataframe():
    class StubProvider:
        def __init__(self):
            self.calls = 0

        def fetch_bars(self, instrument, timeframe, start_time=None, end_time=None, limit=None, include_partial_bar=False):
            self.calls += 1
            if self.calls == 1:
                return pd.DataFrame(
                    [
                        {"time": pd.Timestamp("2026-01-01T00:00:00Z"), "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
                        {"time": pd.Timestamp("2026-01-01T00:01:00Z"), "open": 2, "high": 3, "low": 1.5, "close": 2.5, "volume": 11},
                    ]
                )
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    loader = ProjectXHistoricalDataLoader(StubProvider())
    df = loader.load_range(
        instrument="XAUUSD",
        timeframe="1M",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
        limit_per_request=2,
    )

    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
