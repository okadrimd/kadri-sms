"""Portfolio safety net: kill switch + per-position stop losses.

Two independent layers, checked once per hourly bar by live_trade.py:

1. Portfolio kill switch -- if account equity falls KILL_SWITCH_DD below its
   high-water mark, everything is liquidated and trading halts. The halt is
   sticky (survives restarts) and must be cleared manually:

       python risk.py status   # show HWM, drawdown, halt state, cooldowns
       python risk.py reset    # clear the halt and re-arm from current state

2. Per-position stop loss -- any position whose unrealized loss exceeds
   POSITION_STOP_LOSS is closed, and the ticker is locked out until the next
   trading day so the model cannot immediately re-enter the same falling
   knife.

Limitation: checks happen on the hourly loop, so an overnight gap can exit
worse than the stop level (the exit fills at the next open, not at the stop).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date

import config

log = logging.getLogger("risk")

STATE_PATH = os.path.join(config.BASE_DIR, "risk_state.json")


class RiskManager:
    def __init__(self, state_path: str = STATE_PATH):
        self.state_path = state_path
        self.state = self._load()

    # -- state persistence -------------------------------------------------
    def _load(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as fh:
                return json.load(fh)
        return {"high_water_mark": None, "halted": False, "cooldowns": {}}

    def _save(self) -> None:
        with open(self.state_path, "w") as fh:
            json.dump(self.state, fh, indent=2)

    # -- portfolio kill switch ---------------------------------------------
    @property
    def halted(self) -> bool:
        return bool(self.state.get("halted"))

    def check_kill_switch(self, equity: float) -> bool:
        """Update the high-water mark; return True if the kill switch trips.

        The caller is responsible for liquidating when this returns True.
        """
        hwm = self.state.get("high_water_mark") or equity
        hwm = max(hwm, equity)
        self.state["high_water_mark"] = hwm
        drawdown = equity / hwm - 1.0
        if drawdown <= -config.KILL_SWITCH_DD:
            self.state["halted"] = True
            self.state["halted_reason"] = (
                f"equity ${equity:,.0f} is {drawdown:.1%} below "
                f"high-water mark ${hwm:,.0f}"
            )
            log.error("KILL SWITCH TRIPPED: %s", self.state["halted_reason"])
        self._save()
        return self.halted

    # -- per-position stops -------------------------------------------------
    def stop_loss_hit(self, symbol: str, unrealized_plpc: float) -> bool:
        """Register a stop-out (with next-day cooldown) if the loss is deep."""
        if unrealized_plpc <= -config.POSITION_STOP_LOSS:
            self.state["cooldowns"][symbol] = date.today().isoformat()
            self._save()
            log.warning(
                "STOP LOSS: %s unrealized %.1f%% breaches -%.0f%% -> closing, "
                "locked out until next trading day",
                symbol, unrealized_plpc * 100, config.POSITION_STOP_LOSS * 100,
            )
            return True
        return False

    def in_cooldown(self, symbol: str) -> bool:
        """True while the ticker is locked out (same day as its stop-out)."""
        stamp = self.state.get("cooldowns", {}).get(symbol)
        return stamp == date.today().isoformat()

    # -- manual controls -----------------------------------------------------
    def reset(self) -> None:
        self.state = {"high_water_mark": None, "halted": False, "cooldowns": {}}
        self._save()


def _main() -> None:
    import sys

    rm = RiskManager()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "reset":
        rm.reset()
        print("Risk state cleared: halt lifted, high-water mark re-arms on "
              "the next bar.")
    elif cmd == "status":
        s = rm.state
        print(f"halted          : {s.get('halted')}"
              + (f"  ({s.get('halted_reason')})" if s.get("halted") else ""))
        hwm = s.get("high_water_mark")
        print(f"high-water mark : {'$%s' % format(hwm, ',.0f') if hwm else 'not set'}")
        print(f"cooldowns       : {s.get('cooldowns') or 'none'}")
        print(f"kill switch     : -{config.KILL_SWITCH_DD:.0%} from HWM")
        print(f"position stop   : -{config.POSITION_STOP_LOSS:.0%} from entry")
    else:
        print("usage: python risk.py [status|reset]")


if __name__ == "__main__":
    _main()
