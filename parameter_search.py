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


def main():
    # パラメータグリッド定義
    grid = {
        "SHORT_WINDOW": [3, 5, 7, 9],
        "LONG_WINDOW": [12, 16, 20, 24],
        "MR_RSI_LOWER": [30, 35, 40],
        "MR_RSI_UPPER": [60, 65, 70],
        "STOP_LOSS_PERCENT": [1.0, 1.5, 2.0, 2.5],
        "TAKE_PROFIT_PERCENT": [5.0, 8.0, 10.0, 12.0],
    }
    keys = list(grid.keys())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/param_search_{timestamp}.csv"

    # CSVファイルに結果を書き込む
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys + ["balance"])

        # グリッドサーチ実行
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            # 環境変数にパラメータを設定
            for k, v in params.items():
                os.environ[k] = str(v)

            logger.debug(f"Testing params: {params}")
            balance, trades, _ = run_backtest()
            # balanceがNoneの場合はスキップまたは0として扱う
            writer.writerow([params[k] for k in keys] + [balance if balance is not None else 0])

    logger.info(f"探索結果を保存しました: {results_file}")


if __name__ == "__main__":
    main() 