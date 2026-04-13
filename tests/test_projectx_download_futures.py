from __future__ import annotations

from datetime import datetime, timezone

from scripts.download_projectx_gold import _build_windows, _discover_contracts


class _FakeClient:
    def __init__(self, contracts: dict[str, dict]) -> None:
        self._contracts = contracts

    def search_contract_by_id(self, contract_id: str) -> dict:
        contract = self._contracts.get(contract_id)
        if contract is None:
            raise RuntimeError("not found")
        return {"contract": contract}


def _contract(contract_id: str, symbol_id: str, name: str) -> dict:
    return {"id": contract_id, "symbolId": symbol_id, "name": name}


def test_discover_contracts_uses_api_visible_mgc_months() -> None:
    client = _FakeClient(
        {
            "CON.F.US.MGC.G24": _contract("CON.F.US.MGC.G24", "F.US.MGC", "MGCG4"),
            "CON.F.US.MGC.J24": _contract("CON.F.US.MGC.J24", "F.US.MGC", "MGCJ4"),
            "CON.F.US.MGC.M24": _contract("CON.F.US.MGC.M24", "F.US.MGC", "MGCM4"),
        }
    )

    contracts = _discover_contracts(client, product="MGC", start_year=2024, end_year=2024)

    assert [contract["id"] for contract in contracts] == [
        "CON.F.US.MGC.G24",
        "CON.F.US.MGC.J24",
        "CON.F.US.MGC.M24",
    ]


def test_discover_contracts_supports_monthly_micro_oil_cycle() -> None:
    client = _FakeClient(
        {
            "CON.F.US.MCLE.G24": _contract("CON.F.US.MCLE.G24", "F.US.MCLE", "MCLG4"),
            "CON.F.US.MCLE.H24": _contract("CON.F.US.MCLE.H24", "F.US.MCLE", "MCLH4"),
            "CON.F.US.MCLE.J24": _contract("CON.F.US.MCLE.J24", "F.US.MCLE", "MCLJ4"),
        }
    )

    contracts = _discover_contracts(client, product="MCLE", start_year=2024, end_year=2024)

    assert [contract["id"] for contract in contracts] == [
        "CON.F.US.MCLE.G24",
        "CON.F.US.MCLE.H24",
        "CON.F.US.MCLE.J24",
    ]


def test_build_windows_rolls_forward_without_duplicate_last_contract() -> None:
    contracts = [
        _contract("CON.F.US.MGC.G24", "F.US.MGC", "MGCG4"),
        _contract("CON.F.US.MGC.J24", "F.US.MGC", "MGCJ4"),
        _contract("CON.F.US.MGC.M24", "F.US.MGC", "MGCM4"),
    ]

    windows, earliest_supported_start = _build_windows(
        contracts=contracts,
        requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 6, 15, tzinfo=timezone.utc),
    )

    assert earliest_supported_start == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert [(window.contract_id, window.window_start, window.window_end) for window in windows] == [
        (
            "CON.F.US.MGC.G24",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 2, 1, tzinfo=timezone.utc),
        ),
        (
            "CON.F.US.MGC.J24",
            datetime(2024, 2, 1, tzinfo=timezone.utc),
            datetime(2024, 4, 1, tzinfo=timezone.utc),
        ),
        (
            "CON.F.US.MGC.M24",
            datetime(2024, 4, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 15, tzinfo=timezone.utc),
        ),
    ]
