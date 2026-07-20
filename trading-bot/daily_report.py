"""End-of-day Telegram summary across all configured Alpaca paper accounts.

  python daily_report.py           # send via Telegram (or print if no token)
  python daily_report.py --print   # always print to stdout, never send

Accounts are configured with numbered env vars in .env — account 1 reuses the
trading bot's own keys, and you can add as many more as you like:

  ALPACA_API_KEY / ALPACA_SECRET_KEY            (account 1, the bot's)
  ALPACA_ACCOUNT_NAME_1=Swing Bot               (optional display name)
  ALPACA_API_KEY_2= / ALPACA_SECRET_KEY_2=      (account 2)
  ALPACA_ACCOUNT_NAME_2=...
  ALPACA_API_KEY_3= / ALPACA_SECRET_KEY_3=      (account 3)
  ALPACA_ACCOUNT_NAME_3=...

Telegram delivery needs the okpaca bot's credentials:

  TELEGRAM_BOT_TOKEN=123456:ABC...   (from @BotFather)
  TELEGRAM_CHAT_ID=...               (your chat with the bot; message the bot
                                      once, then visit
                                      api.telegram.org/bot<token>/getUpdates
                                      to read the chat id)

Schedule it ~10 min after the close, Mon-Fri (server on US/Eastern time):

  10 16 * * 1-5 cd /path/to/trading-bot && .venv/bin/python daily_report.py >> report.log 2>&1
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import requests

import config


def configured_accounts() -> list[dict]:
    """Collect numbered ALPACA_API_KEY_N credentials from the environment."""
    accounts = []
    if config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
        accounts.append({
            "name": os.environ.get("ALPACA_ACCOUNT_NAME_1", "Swing Bot"),
            "key": config.ALPACA_API_KEY,
            "secret": config.ALPACA_SECRET_KEY,
        })
    n = 2
    while True:
        key = os.environ.get(f"ALPACA_API_KEY_{n}")
        secret = os.environ.get(f"ALPACA_SECRET_KEY_{n}")
        if not key or not secret:
            break
        accounts.append({
            "name": os.environ.get(f"ALPACA_ACCOUNT_NAME_{n}", f"Account {n}"),
            "key": key,
            "secret": secret,
        })
        n += 1
    return accounts


def summarize_account(name: str, key: str, secret: str) -> str:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    client = TradingClient(key, secret, paper=True)
    acct = client.get_account()
    equity = float(acct.equity)
    last_equity = float(acct.last_equity)  # equity at previous close
    day_pnl = equity - last_equity
    day_pct = day_pnl / last_equity if last_equity else 0.0
    arrow = "🟢" if day_pnl >= 0 else "🔴"

    lines = [
        f"{arrow} *{name}* ({acct.account_number})",
        f"   equity ${equity:,.0f}  day {day_pnl:+,.0f} ({day_pct:+.2%})",
    ]

    positions = client.get_all_positions()
    if positions:
        for p in positions:
            pl = float(p.unrealized_pl)
            plpc = float(p.unrealized_plpc)
            side = "long" if float(p.qty) >= 0 else "short"
            lines.append(
                f"   {p.symbol} {side} {abs(float(p.qty)):g} @ "
                f"${float(p.avg_entry_price):,.2f}  "
                f"P&L {pl:+,.0f} ({plpc:+.1%})"
            )
    else:
        lines.append("   no open positions")

    filled_today = client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.CLOSED,
                         after=datetime.now().replace(hour=0, minute=0, second=0),
                         limit=100)
    )
    fills = [o for o in filled_today if o.filled_at is not None]
    if fills:
        lines.append(f"   {len(fills)} order(s) filled today")
    return "\n".join(lines)


def build_report() -> str:
    accounts = configured_accounts()
    if not accounts:
        return "No Alpaca accounts configured (see daily_report.py docstring)."
    header = f"📊 *Paper trading close report* — {datetime.now():%a %b %d, %Y}"
    body = []
    for acct in accounts:
        try:
            body.append(summarize_account(acct["name"], acct["key"], acct["secret"]))
        except Exception as exc:  # one bad account shouldn't kill the report
            body.append(f"⚠️ *{acct['name']}*: failed to fetch ({exc})")
    return "\n\n".join([header] + body)


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=30,
    )
    resp.raise_for_status()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", dest="print_only",
                        help="print the report instead of sending it")
    args = parser.parse_args()

    report = build_report()
    if not args.print_only and send_telegram(report):
        print("report sent via Telegram")
    else:
        if not args.print_only:
            print("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set; printing:\n")
        print(report)


if __name__ == "__main__":
    main()
