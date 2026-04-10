"""ProjectX REST client and provider implementations."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd
import requests

from src.config.models import InstrumentsConfig, ProjectXConfig

from .base import (
    AccountProvider,
    AccountSnapshot,
    ContractSpec,
    ExecutionProvider,
    ExecutionResult,
    MarketDataProvider,
    OrderSnapshot,
    PositionSnapshot,
)

_ORDER_TYPE_LIMIT = 1
_ORDER_TYPE_MARKET = 2
_ORDER_TYPE_STOP = 4
_SIDE_BUY = 0
_SIDE_SELL = 1

_PROJECTX_UNITS = {
    "1M": (2, 1, timedelta(minutes=1)),
    "5M": (2, 5, timedelta(minutes=5)),
    "15M": (2, 15, timedelta(minutes=15)),
    "1H": (3, 1, timedelta(hours=1)),
    "4H": (3, 4, timedelta(hours=4)),
}


class ProjectXApiError(RuntimeError):
    def __init__(self, message: str, error_code: int | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class ProjectXClient:
    """Thin authenticated REST client for the ProjectX Gateway API."""

    def __init__(
        self,
        base_url: str,
        username: str,
        api_key: str,
        timeout_seconds: float = 30.0,
        token_refresh_buffer_seconds: int = 300,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.token_refresh_buffer_seconds = token_refresh_buffer_seconds
        self.session = session or requests.Session()
        self._token: str | None = None
        self._token_expiry: datetime | None = None
        self._max_retries = 5

    def login(self) -> str:
        payload = {"userName": self.username, "apiKey": self.api_key}
        data = self._post("/api/Auth/loginKey", payload, authenticate=False)
        token = data.get("token")
        if not token:
            raise ProjectXApiError("ProjectX loginKey did not return a token")
        self._token = str(token)
        self._token_expiry = datetime.now(timezone.utc) + timedelta(hours=24)
        return self._token

    def validate(self) -> str:
        if self._token is None:
            return self.login()
        headers = {
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
        }
        response = self.session.post(
            f"{self.base_url}/api/Auth/validate",
            json={},
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("success", False):
            raise ProjectXApiError(
                data.get("errorMessage") or "ProjectX validate failed",
                error_code=data.get("errorCode"),
            )
        new_token = data.get("newToken") or data.get("token")
        if not new_token:
            raise ProjectXApiError("ProjectX validate did not return a refreshed token")
        self._token = str(new_token)
        self._token_expiry = datetime.now(timezone.utc) + timedelta(hours=24)
        return self._token

    def ensure_token(self) -> str:
        now = datetime.now(timezone.utc)
        if self._token is None or self._token_expiry is None:
            return self.login()
        if now >= self._token_expiry - timedelta(seconds=self.token_refresh_buffer_seconds):
            return self.validate()
        return self._token

    def search_accounts(self, only_active_accounts: bool = True) -> dict:
        return self._post("/api/Account/search", {"onlyActiveAccounts": only_active_accounts})

    def search_contracts(self, search_text: str, live: bool) -> dict:
        return self._post("/api/Contract/search", {"searchText": search_text, "live": live})

    def search_contract_by_id(self, contract_id: str) -> dict:
        return self._post("/api/Contract/searchById", {"contractId": contract_id})

    def available_contracts(self, live: bool) -> dict:
        return self._post("/api/Contract/available", {"live": live})

    def retrieve_bars(
        self,
        contract_id: str,
        live: bool,
        start_time: datetime | None,
        end_time: datetime | None,
        unit: int,
        unit_number: int,
        limit: int,
        include_partial_bar: bool,
    ) -> dict:
        payload = {
            "contractId": contract_id,
            "live": live,
            "startTime": _iso_or_none(start_time),
            "endTime": _iso_or_none(end_time),
            "unit": unit,
            "unitNumber": unit_number,
            "limit": limit,
            "includePartialBar": include_partial_bar,
        }
        return self._post("/api/History/retrieveBars", payload)

    def search_open_orders(self, account_id: int) -> dict:
        return self._post("/api/Order/searchOpen", {"accountId": account_id})

    def place_order(self, payload: dict) -> dict:
        return self._post("/api/Order/place", payload)

    def cancel_order(self, account_id: int, order_id: int | str) -> dict:
        return self._post("/api/Order/cancel", {"accountId": account_id, "orderId": order_id})

    def modify_order(
        self,
        account_id: int,
        order_id: int | str,
        size: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
    ) -> dict:
        return self._post(
            "/api/Order/modify",
            {
                "accountId": account_id,
                "orderId": order_id,
                "size": size,
                "limitPrice": limit_price,
                "stopPrice": stop_price,
                "trailPrice": trail_price,
            },
        )

    def search_open_positions(self, account_id: int) -> dict:
        return self._post("/api/Position/searchOpen", {"accountId": account_id})

    def close_contract(self, account_id: int, contract_id: str) -> dict:
        return self._post(
            "/api/Position/closeContract",
            {"accountId": account_id, "contractId": contract_id},
        )

    def partial_close_contract(self, account_id: int, contract_id: str, size: int) -> dict:
        return self._post(
            "/api/Position/partialCloseContract",
            {"accountId": account_id, "contractId": contract_id, "size": size},
        )

    def _post(self, path: str, payload: dict, authenticate: bool = True) -> dict:
        headers = {"accept": "text/plain", "Content-Type": "application/json"}
        if authenticate:
            headers["Authorization"] = f"Bearer {self.ensure_token()}"

        for attempt in range(self._max_retries):
            response = self.session.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            if response.status_code == 401 and authenticate:
                self.login()
                headers["Authorization"] = f"Bearer {self._token}"
                continue
            if response.status_code == 429:
                wait_seconds = min(30, 2 ** attempt)
                time.sleep(wait_seconds)
                continue
            if response.status_code >= 500 and attempt < self._max_retries - 1:
                time.sleep(min(10, 2 ** attempt))
                continue
            break

        response.raise_for_status()
        data = response.json()
        if not data.get("success", False):
            raise ProjectXApiError(
                data.get("errorMessage") or f"ProjectX request failed for {path}",
                error_code=data.get("errorCode"),
            )
        return data


class ProjectXMarketDataProvider(MarketDataProvider):
    def __init__(
        self,
        client: ProjectXClient,
        config: ProjectXConfig,
        instruments: InstrumentsConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.instruments = instruments or InstrumentsConfig()
        self._contract_cache: dict[str, ContractSpec] = {}

    def get_multi_tf_data(
        self,
        instrument: str,
        count: int = 500,
        include_partial_bar: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for timeframe in ("1M", "5M", "15M", "1H", "4H"):
            result[timeframe] = self.fetch_bars(
                instrument=instrument,
                timeframe=timeframe,
                limit=count,
                include_partial_bar=include_partial_bar,
            )
        return result

    def fetch_bars(
        self,
        instrument: str,
        timeframe: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        include_partial_bar: bool = False,
    ) -> pd.DataFrame:
        tf = timeframe.upper()
        if tf not in _PROJECTX_UNITS:
            raise ValueError(f"Unsupported ProjectX timeframe: {timeframe}")

        spec = self.get_contract_spec(instrument)
        unit, unit_number, _ = _PROJECTX_UNITS[tf]
        data = self.client.retrieve_bars(
            contract_id=spec.contract_id,
            live=self.config.live,
            start_time=start_time,
            end_time=end_time,
            unit=unit,
            unit_number=unit_number,
            limit=limit or self.config.bar_limit,
            include_partial_bar=include_partial_bar,
        )
        bars = data.get("bars", [])
        if not bars:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        rows = [
            {
                "time": _parse_bar_time(bar["t"]),
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": float(bar.get("v", 0)),
            }
            for bar in bars
        ]
        return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    def get_tick(self, instrument: str) -> Dict[str, Any]:
        df = self.fetch_bars(
            instrument=instrument,
            timeframe="1M",
            limit=1,
            include_partial_bar=True,
        )
        if df.empty:
            return {}
        row = df.iloc[-1]
        price = float(row["close"])
        return {
            "bid": price,
            "ask": price,
            "spread": 0.0,
            "time": pd.Timestamp(row["time"], tz="UTC"),
        }

    def get_contract_spec(self, instrument: str) -> ContractSpec:
        if instrument in self._contract_cache:
            return self._contract_cache[instrument]

        override = self.instruments.get(instrument)
        contract_id = None
        symbol_id = None
        if override is not None and override.contract_id:
            contract_id = override.contract_id
        elif self.config.default_contract_id:
            contract_id = self.config.default_contract_id
        if override is not None and override.symbol_id:
            symbol_id = override.symbol_id
        elif self.config.default_symbol_id:
            symbol_id = self.config.default_symbol_id

        contract: dict[str, Any] | None = None
        if contract_id:
            contract = self.client.search_contract_by_id(contract_id).get("contract")
        elif symbol_id:
            contracts = self.client.available_contracts(live=self.config.live).get("contracts", [])
            matches = [c for c in contracts if c.get("symbolId") == symbol_id]
            if matches:
                contract = next((c for c in matches if c.get("activeContract")), matches[0])
        else:
            contracts = self.client.search_contracts(search_text=instrument, live=self.config.live).get("contracts", [])
            if contracts:
                contract = next((c for c in contracts if c.get("activeContract")), contracts[0])

        if contract is None:
            raise ProjectXApiError(f"Could not resolve ProjectX contract for instrument '{instrument}'")

        tick_size = _coalesce_float(
            getattr(override, "tick_size", None) if override else None,
            contract.get("tickSize"),
        )
        tick_value = _coalesce_float(
            getattr(override, "tick_value", None) if override else None,
            contract.get("tickValue"),
        )
        point_value = tick_value / tick_size if tick_size and tick_value else None

        spec = ContractSpec(
            contract_id=contract["id"],
            symbol_id=(
                getattr(override, "symbol_id", None)
                if override and override.symbol_id
                else contract.get("symbolId") or self.config.default_symbol_id
            ),
            name=contract.get("name"),
            description=contract.get("description"),
            tick_size=tick_size,
            tick_value=tick_value,
            point_value=point_value,
            default_quantity=getattr(override, "default_quantity", None) if override else None,
            provider="projectx",
            raw=contract,
        )
        self._contract_cache[instrument] = spec
        return spec


class ProjectXExecutionProvider(ExecutionProvider):
    def __init__(
        self,
        client: ProjectXClient,
        market_data_provider: ProjectXMarketDataProvider,
        account_id: int,
    ) -> None:
        self.client = client
        self.market_data_provider = market_data_provider
        self.account_id = account_id

    def place_market_order(
        self,
        instrument: str,
        direction: str,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        comment: str = "",
    ) -> ExecutionResult:
        spec = self.market_data_provider.get_contract_spec(instrument)
        tick_size = float(spec.tick_size or 0.0)
        current_price = self.market_data_provider.get_tick(instrument).get("ask")
        payload = {
            "accountId": self.account_id,
            "contractId": spec.contract_id,
            "type": _ORDER_TYPE_MARKET,
            "side": _SIDE_BUY if direction == "long" else _SIDE_SELL,
            "size": max(1, int(round(quantity))),
            "limitPrice": None,
            "stopPrice": None,
            "trailPrice": None,
            "customTag": comment or None,
            "stopLossBracket": (
                {"ticks": _price_distance_to_ticks(stop_loss, current_price, tick_size), "type": _ORDER_TYPE_STOP}
                if stop_loss is not None and tick_size > 0 and current_price is not None
                else None
            ),
            "takeProfitBracket": (
                {"ticks": _price_distance_to_ticks(take_profit, current_price, tick_size), "type": _ORDER_TYPE_LIMIT}
                if take_profit is not None and tick_size > 0 and current_price is not None
                else None
            ),
        }
        response = self.client.place_order(payload)
        return ExecutionResult(
            success=True,
            order_id=response.get("orderId"),
            fill_price=current_price,
            quantity=float(payload["size"]),
            error_code=response.get("errorCode"),
            raw=response,
        )

    def place_limit_order(
        self,
        instrument: str,
        direction: str,
        quantity: float,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        comment: str = "",
    ) -> ExecutionResult:
        spec = self.market_data_provider.get_contract_spec(instrument)
        tick_size = float(spec.tick_size or 0.0)
        payload = {
            "accountId": self.account_id,
            "contractId": spec.contract_id,
            "type": _ORDER_TYPE_LIMIT,
            "side": _SIDE_BUY if direction == "long" else _SIDE_SELL,
            "size": max(1, int(round(quantity))),
            "limitPrice": price,
            "stopPrice": None,
            "trailPrice": None,
            "customTag": comment or None,
            "stopLossBracket": (
                {"ticks": _price_distance_to_ticks(stop_loss, price, tick_size), "type": _ORDER_TYPE_STOP}
                if stop_loss is not None and tick_size > 0
                else None
            ),
            "takeProfitBracket": (
                {"ticks": _price_distance_to_ticks(take_profit, price, tick_size), "type": _ORDER_TYPE_LIMIT}
                if take_profit is not None and tick_size > 0
                else None
            ),
        }
        response = self.client.place_order(payload)
        return ExecutionResult(
            success=True,
            order_id=response.get("orderId"),
            fill_price=price,
            quantity=float(payload["size"]),
            error_code=response.get("errorCode"),
            raw=response,
        )

    def modify_order(
        self,
        order_id: int | str,
        quantity: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
    ) -> bool:
        self.client.modify_order(
            account_id=self.account_id,
            order_id=order_id,
            size=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            trail_price=trail_price,
        )
        return True

    def cancel_order(self, order_id: int | str) -> bool:
        self.client.cancel_order(account_id=self.account_id, order_id=order_id)
        return True

    def close_position(self, instrument: str, quantity: float | None = None) -> bool:
        spec = self.market_data_provider.get_contract_spec(instrument)
        if quantity is None:
            self.client.close_contract(account_id=self.account_id, contract_id=spec.contract_id)
        else:
            self.client.partial_close_contract(
                account_id=self.account_id,
                contract_id=spec.contract_id,
                size=max(1, int(round(quantity))),
            )
        return True

    def partial_close_position(self, instrument: str, quantity: float) -> bool:
        return self.close_position(instrument, quantity=quantity)


class ProjectXAccountProvider(AccountProvider):
    def __init__(
        self,
        client: ProjectXClient,
        market_data_provider: ProjectXMarketDataProvider,
        account_id: int,
    ) -> None:
        self.client = client
        self.market_data_provider = market_data_provider
        self.account_id = account_id

    def get_account_info(self) -> AccountSnapshot | None:
        accounts = self.client.search_accounts(only_active_accounts=True).get("accounts", [])
        account = next((a for a in accounts if int(a.get("id")) == int(self.account_id)), None)
        if account is None:
            return None
        balance = float(account.get("balance", 0.0) or 0.0)
        return AccountSnapshot(
            account_id=int(account["id"]),
            balance=balance,
            equity=balance,
            free_margin=balance,
            can_trade=bool(account.get("canTrade", True)),
            raw=account,
        )

    def get_positions(self, instrument: str | None = None) -> list[PositionSnapshot]:
        rows = self.client.search_open_positions(account_id=self.account_id).get("positions", [])
        result: list[PositionSnapshot] = []
        for row in rows:
            contract_id = str(row.get("contractId"))
            if instrument is not None:
                try:
                    spec = self.market_data_provider.get_contract_spec(instrument)
                    if spec.contract_id != contract_id:
                        continue
                except Exception:
                    continue
            side = "long" if int(row.get("type", 0)) == _SIDE_BUY else "short"
            result.append(
                PositionSnapshot(
                    position_id=row.get("id"),
                    account_id=row.get("accountId"),
                    contract_id=contract_id,
                    instrument=instrument or contract_id,
                    direction=side,
                    quantity=float(row.get("size", 0)),
                    entry_price=float(row.get("averagePrice", 0.0) or 0.0),
                    time=_parse_ts(row.get("creationTimestamp")),
                    raw=row,
                )
            )
        return result

    def get_open_orders(self, instrument: str | None = None) -> list[OrderSnapshot]:
        rows = self.client.search_open_orders(account_id=self.account_id).get("orders", [])
        result: list[OrderSnapshot] = []
        for row in rows:
            contract_id = str(row.get("contractId"))
            if instrument is not None:
                try:
                    spec = self.market_data_provider.get_contract_spec(instrument)
                    if spec.contract_id != contract_id:
                        continue
                except Exception:
                    continue
            result.append(
                OrderSnapshot(
                    order_id=row.get("id"),
                    account_id=row.get("accountId"),
                    contract_id=contract_id,
                    instrument=instrument or contract_id,
                    side="long" if int(row.get("side", 0)) == _SIDE_BUY else "short",
                    order_type=str(row.get("type")),
                    quantity=float(row.get("size", 0)),
                    status=row.get("status"),
                    limit_price=_coalesce_float(row.get("limitPrice")),
                    stop_price=_coalesce_float(row.get("stopPrice")),
                    filled_price=_coalesce_float(row.get("filledPrice")),
                    raw=row,
                )
            )
        return result


class ProjectXHistoricalDataLoader:
    """Fetch historical ProjectX bars in chunks and return a backtest-ready DataFrame."""

    def __init__(self, market_data_provider: ProjectXMarketDataProvider) -> None:
        self.market_data_provider = market_data_provider

    def load_range(
        self,
        instrument: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        limit_per_request: int = 20_000,
    ) -> pd.DataFrame:
        tf = timeframe.upper()
        if tf not in _PROJECTX_UNITS:
            raise ValueError(f"Unsupported timeframe for ProjectX historical load: {timeframe}")

        _, _, step = _PROJECTX_UNITS[tf]
        chunk_span = step * limit_per_request
        current_start = start_time
        chunks: list[pd.DataFrame] = []

        while current_start < end_time:
            current_end = min(end_time, current_start + chunk_span)
            chunk = self.market_data_provider.fetch_bars(
                instrument=instrument,
                timeframe=tf,
                start_time=current_start,
                end_time=current_end,
                limit=limit_per_request,
                include_partial_bar=False,
            )
            if not chunk.empty:
                chunks.append(chunk)
                latest = pd.Timestamp(chunk["time"].iloc[-1]).to_pydatetime()
                current_start = latest + step
            else:
                current_start = current_end + step

        if not chunks:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
        df = df.set_index(pd.DatetimeIndex(df["time"], tz="UTC")).drop(columns=["time"])
        return df


def build_projectx_stack(
    config: ProjectXConfig,
    instruments: InstrumentsConfig | None = None,
    username: str | None = None,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> tuple[ProjectXClient, ProjectXMarketDataProvider, ProjectXExecutionProvider | None, ProjectXAccountProvider | None]:
    username = username or os.getenv(config.username_env)
    api_key = api_key or os.getenv(config.api_key_env)
    if not username or not api_key:
        raise ProjectXApiError(
            "ProjectX credentials are not configured. "
            f"Expected env vars {config.username_env} and {config.api_key_env}."
        )

    client = ProjectXClient(
        base_url=config.api_base_url,
        username=username,
        api_key=api_key,
        timeout_seconds=config.request_timeout_seconds,
        token_refresh_buffer_seconds=config.token_refresh_buffer_seconds,
        session=session,
    )
    market = ProjectXMarketDataProvider(client=client, config=config, instruments=instruments)

    execution = None
    account = None
    if config.account_id is not None:
        execution = ProjectXExecutionProvider(
            client=client,
            market_data_provider=market,
            account_id=config.account_id,
        )
        account = ProjectXAccountProvider(
            client=client,
            market_data_provider=market,
            account_id=config.account_id,
        )
    return client, market, execution, account


def _iso_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return pd.Timestamp(value, tz="UTC").to_pydatetime()


def _parse_bar_time(value: Any) -> pd.Timestamp:
    if isinstance(value, (int, float)):
        # ProjectX timestamps may be epoch milliseconds or seconds depending on client.
        unit = "ms" if value > 10_000_000_000 else "s"
        return pd.to_datetime(value, unit=unit, utc=True)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _coalesce_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _price_distance_to_ticks(target_price: float | None, reference_price: float | None, tick_size: float) -> int | None:
    if target_price is None or reference_price is None or tick_size <= 0:
        return None
    ticks = int(round(abs(float(target_price) - float(reference_price)) / tick_size))
    return max(1, ticks)
