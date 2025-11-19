#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optunaによるパラメータ最適化スクリプト
バックテストをベイズ最適化で実行し、年利改善のための最適パラメータを探索します。
"""
import os
import optuna
from datetime import datetime
from loguru import logger
from backtest_runner import run_backtest


import argparse
import sys

def objective(trial):
    # パラメータ範囲の指定
    short_window = trial.suggest_int("SHORT_WINDOW", 3, 15, step=2)
    long_window = trial.suggest_int("LONG_WINDOW", 20, 60, step=5)
    mr_rsi_lower = trial.suggest_int("MR_RSI_LOWER", 20, 40, step=5)
    mr_rsi_upper = trial.suggest_int("MR_RSI_UPPER", 60, 80, step=5)
    stop_loss_percent = trial.suggest_float("STOP_LOSS_PERCENT", 0.5, 5.0, step=0.5)
    take_profit_percent = trial.suggest_float("TAKE_PROFIT_PERCENT", 1.0, 15.0, step=1.0)
    max_consecutive_losses = trial.suggest_int("MAX_CONSECUTIVE_LOSSES", 3, 10, step=1)
    
    # Trend戦略パラメータ
    trend_adx_threshold = trial.suggest_int("TREND_ADX_THRESHOLD", 15, 35, step=5)
    trend_short_window = trial.suggest_int("TREND_SHORT_WINDOW", 5, 15, step=2)
    trend_long_window = trial.suggest_int("TREND_LONG_WINDOW", 20, 50, step=5)

    # 環境変数に設定
    os.environ["SHORT_WINDOW"] = str(short_window)
    os.environ["LONG_WINDOW"] = str(long_window)
    os.environ["MR_RSI_LOWER"] = str(mr_rsi_lower)
    os.environ["MR_RSI_UPPER"] = str(mr_rsi_upper)
    os.environ["STOP_LOSS_PERCENT"] = str(stop_loss_percent)
    os.environ["TAKE_PROFIT_PERCENT"] = str(take_profit_percent)
    os.environ["MAX_CONSECUTIVE_LOSSES"] = str(max_consecutive_losses)
    
    # Trend戦略用環境変数
    os.environ["TREND_ADX_THRESHOLD"] = str(trend_adx_threshold)
    os.environ["ENABLE_TREND"] = "true" # 最適化中は強制有効化

    try:
        balance, _, _ = run_backtest()
        return balance if balance is not None else 0.0
    except Exception as e:
        logger.error(f"Backtest failed with params: {trial.params}, error: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Optuna Parameter Optimization")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--study-name", type=str, default=None, help="Study name")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"optuna_{timestamp}"

    # Bayesian Optimization 用のStudyを作成
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        study_name=study_name
    )
    
    logger.info(f"Starting optimization with {args.trials} trials...")
    study.optimize(objective, n_trials=args.trials)

    # 結果の保存
    os.makedirs("results", exist_ok=True)
    df = study.trials_dataframe()
    results_file = f"results/optuna_results_{timestamp}.csv"
    df.to_csv(results_file, index=False)

    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best balance: {study.best_value}")
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main() 