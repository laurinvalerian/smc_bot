import argparse
from backtester import run_backtest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data", help="Ordner mit allen CSV-Dateien")
    parser.add_argument(
        "--last_years",
        type=int,
        default=None,
        help="Nur letzte X Jahre testen (z.B. 1 für 2025 oder letztes Jahr)",
    )
    parser.add_argument(
        "--pair",
        default=None,
        help="Optional: nur ein Pair testen (z.B. AUDUSD_2018)",
    )  # reserved for future per-pair filtering
    args = parser.parse_args()

    print("🚀 Starte SMC Backtest...")
    raw = run_backtest(args.folder, args.last_years)
    summary = raw["summary"]

    results = {
        "total_trades": summary["total_trades"],
        "winrate": summary["overall_winrate"] * 100,
        "profit_factor": summary["profit_factor"],
        "net_profit": summary["total_return_pct"],
        "max_dd": summary["max_dd"],
    }

    print("\n" + "=" * 60)
    print("BACKTEST ERGEBNISSE")
    print("=" * 60)
    print(f"Gesamt Trades: {results['total_trades']}")
    print(f"Winrate: {results['winrate']:.1f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Net Profit: {results['net_profit']:.2f}%")
    print(f"Max Drawdown: {results['max_dd']:.2f}%")
    print("=" * 60)
