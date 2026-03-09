"""
main_optimizer.py
=================

Entry point for the SMC Grid-Search Optimizer.

Usage
-----
  python main_optimizer.py

Outputs
-------
  optimization_results.csv  – all parameter combinations with metrics
  top_15.txt                – top 15 combinations sorted by Score
"""

import multiprocessing

from optimizer import run_optimization

if __name__ == "__main__":
    # Required on Windows; harmless on macOS / Linux
    multiprocessing.freeze_support()
    run_optimization()
