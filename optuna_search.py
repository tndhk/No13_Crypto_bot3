#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optunaによるパラメータ最適化スクリプト
バックテストをベイズ最適化で実行し、年利改善のための最適パラメータを探索します。
"""
import os
import optuna
from datetime import datetime
from backtest_runner import run_backtest


def objective(trial):
    # パラメータ範囲の指定
    short_window = trial.suggest_int("SHORT_WINDOW", 3, 9, step=2)
    long_window = trial.suggest_int("LONG_WINDOW", 12, 24, step=4)
    mr_rsi_lower = trial.suggest_int("MR_RSI_LOWER", 30, 40, step=5)
    mr_rsi_upper = trial.suggest_int("MR_RSI_UPPER", 60, 70, step=5)
    stop_loss_percent = trial.suggest_float("STOP_LOSS_PERCENT", 1.0, 2.5, step=0.5)
    take_profit_percent = trial.suggest_float("TAKE_PROFIT_PERCENT", 5.0, 12.0, step=2.5)

    # 環境変数に設定
    os.environ["SHORT_WINDOW"] = str(short_window)
    os.environ["LONG_WINDOW"] = str(long_window)
    os.environ["MR_RSI_LOWER"] = str(mr_rsi_lower)
    os.environ["MR_RSI_UPPER"] = str(mr_rsi_upper)
    os.environ["STOP_LOSS_PERCENT"] = str(stop_loss_percent)
    os.environ["TAKE_PROFIT_PERCENT"] = str(take_profit_percent)

    balance, _, _ = run_backtest()
    return balance if balance is not None else 0.0


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Bayesian Optimization 用のStudyを作成
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        study_name=f"optuna_{timestamp}"
    )
    # 試行回数 (必要に応じて調整可能)
    study.optimize(objective, n_trials=50)

    # 結果の保存
    os.makedirs("results", exist_ok=True)
    df = study.trials_dataframe()
    results_file = f"results/optuna_results_{timestamp}.csv"
    df.to_csv(results_file, index=False)

    print("Best parameters:", study.best_params)
    print("Best balance:", study.best_value)
    print("Results saved to", results_file)


if __name__ == "__main__":
    main() 