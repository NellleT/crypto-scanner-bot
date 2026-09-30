"""Real orders are impossible by construction — and must stay that way.

The bot reads public market data and builds order payloads for display; it has
no code that places, cancels or funds an order, and its exchange client never
holds credentials. These tests fail if either stops being true.
"""

from __future__ import annotations

import re

from scanner.config import PROJECT_ROOT
from scanner.exchange import MarketDataClient

FORBIDDEN = re.compile(
    r"\.create_(?:order|market_\w+|limit_\w+|orders)\s*\("
    r"|\.cancel_(?:order|all_orders)\s*\("
    r"|\.(?:fetch_balance|withdraw|transfer)\s*\("
    r"|\.private_\w+\s*\("
    r"|['\"](?:apiKey|secret|password|privateKey)['\"]\s*:"
)


def sources():
    yield PROJECT_ROOT / "main.py"
    yield from sorted((PROJECT_ROOT / "scanner").glob("*.py"))


def test_no_code_can_place_cancel_or_fund_an_order() -> None:
    offenders = [
        f"{path.name}:{number}: {line.strip()}"
        for path in sources()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if FORBIDDEN.search(line)
    ]
    assert offenders == []


def test_the_exchange_client_holds_no_credentials() -> None:
    client = MarketDataClient("binance", public_api_url="https://data-api.binance.vision/api/v3")
    assert not client._exchange.apiKey and not client._exchange.secret
    assert client._exchange.urls["api"]["public"] == "https://data-api.binance.vision/api/v3"
    assert client._exchange.options["fetchMarkets"] == ["spot"]
