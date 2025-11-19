#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
パラメータ最適化用スクリプト
バックテストをグリッドサーチで実行し、年利改善のための最適パラメータを探索します。
"""
import os
import csv
import itertools
from datetime import datetime
from loguru import logger
from backtest_runner import run_backtest


import argparse

def main():
    parser = argparse.ArgumentParser(description="Grid Search Parameter Optimization")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    # パラメータグリッド定義
    # 注意: 組み合わせが爆発しないように注意
    grid = {
        "SHORT_WINDOW": [5, 9, 13],
        "LONG_WINDOW": [20, 40, 60],
        "MR_RSI_LOWER": [30, 35],
        "MR_RSI_UPPER": [65, 70],
        "STOP_LOSS_PERCENT": [1.5, 3.0],
        "TAKE_PROFIT_PERCENT": [5.0, 10.0],
    }
    keys = list(grid.keys())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/param_search_{timestamp}.csv"

    # Calculate total combinations
    total_combinations = 1
    for k in keys:
        total_combinations *= len(grid[k])
    
    logger.info(f"Starting grid search with {total_combinations} combinations...")

    # CSVファイルに結果を書き込む
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys + ["balance"])

        # グリッドサーチ実行
        for i, combo in enumerate(itertools.product(*(grid[k] for k in keys))):
            params = dict(zip(keys, combo))
            # 環境変数にパラメータを設定
            for k, v in params.items():
                os.environ[k] = str(v)

            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total_combinations}")

            try:
                balance, trades, _ = run_backtest()
                # balanceがNoneの場合はスキップまたは0として扱う
                writer.writerow([params[k] for k in keys] + [balance if balance is not None else 0])
            except Exception as e:
                logger.error(f"Backtest failed with params: {params}, error: {e}")
                writer.writerow([params[k] for k in keys] + [0])

    logger.info(f"探索結果を保存しました: {results_file}")


if __name__ == "__main__":
    main() 