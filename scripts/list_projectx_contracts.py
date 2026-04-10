"""
ProjectX contract discovery helper.

Examples
--------
python scripts/list_projectx_contracts.py --query gold
python scripts/list_projectx_contracts.py --query MGC
python scripts/list_projectx_contracts.py --symbol-id F.US.MGC
python scripts/list_projectx_contracts.py --contract-id CON.F.US.MGC.M26
python scripts/list_projectx_contracts.py --available --live
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_local_env() -> None:
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List/search ProjectX contracts and print the exact IDs needed for config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config directory override.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Free-text contract search, e.g. gold, gc, mgc, mes.",
    )
    parser.add_argument(
        "--symbol-id",
        type=str,
        default=None,
        help="Filter available contracts by symbolId, e.g. F.US.MGC.",
    )
    parser.add_argument(
        "--contract-id",
        type=str,
        default=None,
        help="Fetch one exact contract by id.",
    )
    parser.add_argument(
        "--available",
        action="store_true",
        help="List from the available-contracts endpoint instead of text search.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live=true when querying ProjectX contracts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of contracts to print.",
    )
    parser.add_argument(
        "--projectx-username",
        type=str,
        default=None,
        help="Optional username override. Otherwise uses env/config.",
    )
    parser.add_argument(
        "--projectx-api-key",
        type=str,
        default=None,
        help="Optional API key override. Otherwise uses env/config.",
    )
    return parser


def _print_contract(contract: dict) -> None:
    print(f"name:        {contract.get('name')}")
    print(f"contract_id: {contract.get('id')}")
    print(f"symbol_id:   {contract.get('symbolId')}")
    print(f"tick_size:   {contract.get('tickSize')}")
    print(f"tick_value:  {contract.get('tickValue')}")
    print(f"active:      {contract.get('activeContract')}")
    print(f"description: {contract.get('description')}")
    print("-" * 72)


def main() -> int:
    _load_local_env()
    args = _build_parser().parse_args()

    try:
        from src.config.loader import ConfigLoader
        from src.providers import ProjectXApiError, build_projectx_stack

        app_config = ConfigLoader(config_dir=args.config).load()
        cfg = app_config.provider.projectx.model_copy(deep=True)
        cfg.live = bool(args.live)

        client, _, _, _ = build_projectx_stack(
            config=cfg,
            instruments=app_config.instruments,
            username=args.projectx_username,
            api_key=args.projectx_api_key,
        )

        if args.contract_id:
            response = client.search_contract_by_id(args.contract_id)
            contract = response.get("contract")
            if not contract:
                print("No contract returned.")
                return 1
            _print_contract(contract)
            return 0

        if args.available or args.symbol_id:
            response = client.available_contracts(live=cfg.live)
            contracts = response.get("contracts", [])
            if args.symbol_id:
                contracts = [c for c in contracts if c.get("symbolId") == args.symbol_id]
        else:
            query = args.query or "gold"
            response = client.search_contracts(search_text=query, live=cfg.live)
            contracts = response.get("contracts", [])

        if not contracts:
            print("No contracts found.")
            return 1

        print(f"Found {len(contracts)} contract(s). Showing up to {args.limit}.")
        print("-" * 72)
        for contract in contracts[: max(1, args.limit)]:
            _print_contract(contract)

        active = [c for c in contracts if c.get("activeContract")]
        if active:
            print("Suggested active contract:")
            print(f"contract_id={active[0].get('id')}")
            print(f"symbol_id={active[0].get('symbolId')}")

        return 0

    except ProjectXApiError as exc:
        print(f"ProjectX API error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
