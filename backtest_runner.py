#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mean Reversion強化版バックテストランナー

強化されたMean Reversion戦略でバックテストを実行し、
結果の分析と可視化を行います。
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
import sys
import traceback

# ロギング設定
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/backtest_runner_{time}.log", rotation="1 day", retention="30 days")

def run_backtest(start_time=None, end_time=None, symbol=None, interval=None, output_dir='results'):
    """
    指定された期間でバックテストを実行
    
    Parameters:
    -----------
    start_time : str, optional
        開始時間 (ISO形式: "YYYY-MM-DDTHH:MM:SS")
    end_time : str, optional
        終了時間 (ISO形式: "YYYY-MM-DDTHH:MM:SS")
    symbol : str, optional
        取引ペア (例: "BTCUSDT")
    interval : str, optional
        時間枠 (例: "1h")
    output_dir : str
        出力ディレクトリ
    
    Returns:
    --------
    tuple
        (最終残高, 取引リスト, 残高履歴)
    """
    # 環境変数の設定
    if start_time:
        os.environ["START_TIME"] = start_time
    if end_time:
        os.environ["END_TIME"] = end_time
    if symbol:
        os.environ["SYMBOL"] = symbol
    if interval:
        os.environ["INTERVAL"] = interval
    
    # 必要なディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 強化版ボットのインポート（この段階でインポートして依存関係の問題を避ける）
        from enhanced_trading_bot import EnhancedTradingBot
        
        # 強化版ボットの初期化
        bot = EnhancedTradingBot()
        
        # バックテスト実行
        logger.info("バックテストを実行中...")
        balance, trades = bot.run_backtest()
        
        if balance is not None:
            # 結果の可視化
            logger.info(f"バックテスト完了。残高: {balance:.2f} USDT, 取引数: {len([t for t in trades if t['type'] == 'BUY'])}")
            visualize_results(bot.balance_history, trades, output_dir=output_dir)
            return balance, trades, bot.balance_history
        else:
            logger.error("バックテストが正常に完了しませんでした")
            return None, None, None
    
    except ImportError as e:
        logger.error(f"モジュールのインポートに失敗しました: {e}")
        logger.error(traceback.format_exc())
        return None, None, None
    except Exception as e:
        logger.error(f"バックテスト実行中にエラーが発生しました: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

def visualize_results(balance_history, trades, output_dir="results"):
    """
    バックテスト結果の可視化
    
    Parameters:
    -----------
    balance_history : list
        残高履歴 [(timestamp, balance), ...]
    trades : list
        取引リスト
    output_dir : str
        出力ディレクトリ
    """
    # 出力ディレクトリが空でないことを確認
    if not balance_history or not trades:
        logger.error("可視化するデータがありません")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 残高の推移をプロット
        plt.figure(figsize=(12, 6))
        balance_df = pd.DataFrame(balance_history, columns=["timestamp", "balance"])
        
        # 残高の推移
        plt.subplot(2, 1, 1)
        plt.plot(balance_df["timestamp"], balance_df["balance"])
        plt.title("残高推移")
        plt.ylabel("残高 (USDT)")
        plt.grid(True)
        
        # 取引を残高推移にマーク
        buy_trades = [t for t in trades if t["type"] == "BUY"]
        sell_trades = [t for t in trades if t["type"] == "SELL"]
        
        for trade in buy_trades:
            if "timestamp" not in trade:
                continue
            trade_time = trade["timestamp"]
            # 最も近い残高履歴のエントリを探す
            closest_idx = abs(balance_df["timestamp"] - trade_time).argmin()
            trade_balance = balance_df.iloc[closest_idx]["balance"]
            plt.scatter(trade_time, trade_balance, color="green", marker="^", s=50)
        
        for trade in sell_trades:
            if "timestamp" not in trade:
                continue
            trade_time = trade["timestamp"]
            closest_idx = abs(balance_df["timestamp"] - trade_time).argmin()
            trade_balance = balance_df.iloc[closest_idx]["balance"]
            
            # 利益が出ているかどうかでマーカーの色を変える
            marker_color = "blue" if trade.get("net_profit", 0) > 0 else "red"
            plt.scatter(trade_time, trade_balance, color=marker_color, marker="v", s=50)
        
        # ドローダウンをプロット
        plt.subplot(2, 1, 2)
        balance_df["peak"] = balance_df["balance"].cummax()
        balance_df["drawdown"] = (balance_df["peak"] - balance_df["balance"]) / balance_df["peak"] * 100
        
        plt.plot(balance_df["timestamp"], balance_df["drawdown"])
        plt.title("ドローダウン")
        plt.ylabel("ドローダウン (%)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/balance_history_{timestamp}.png")
        plt.close()
        
        # 戦略別のパフォーマンスを分析
        if trades:
            strategy_performance = analyze_strategy_performance(trades)
            visualize_strategy_performance(strategy_performance, output_dir, timestamp)
        
        logger.info(f"可視化結果を {output_dir} ディレクトリに保存しました")
    
    except Exception as e:
        logger.error(f"可視化中にエラーが発生しました: {e}")
        logger.error(traceback.format_exc())

def analyze_strategy_performance(trades):
    """
    戦略別のパフォーマンスを分析
    
    Parameters:
    -----------
    trades : list
        取引リスト
    
    Returns:
    --------
    dict
        戦略別のパフォーマンス指標
    """
    strategy_performance = {}
    
    # 取引ごとに戦略を特定し、パフォーマンスを集計
    for trade in trades:
        if trade["type"] != "SELL" or "strategy" not in trade:
            continue
        
        strategy = trade["strategy"]
        
        if strategy not in strategy_performance:
            strategy_performance[strategy] = {
                "count": 0,
                "wins": 0,
                "losses": 0,
                "profit": 0,
                "profit_percent": 0,
                "hold_durations": [],
                "profits": []  # 各取引の利益率
            }
        
        performance = strategy_performance[strategy]
        performance["count"] += 1
        
        net_profit = trade.get("net_profit", 0)
        profit_percent = trade.get("profit_percent", 0)
        
        if net_profit > 0:
            performance["wins"] += 1
        else:
            performance["losses"] += 1
            
        performance["profit"] += net_profit
        performance["profit_percent"] += profit_percent
        performance["profits"].append(profit_percent)
        
        if "hold_duration" in trade:
            performance["hold_durations"].append(trade["hold_duration"])
    
    # 追加の指標を計算
    for strategy, performance in strategy_performance.items():
        total_trades = performance["count"]
        if total_trades > 0:
            performance["win_rate"] = performance["wins"] / total_trades * 100
            performance["avg_profit_percent"] = performance["profit_percent"] / total_trades
            performance["avg_hold_duration"] = sum(performance["hold_durations"]) / len(performance["hold_durations"]) if performance["hold_durations"] else 0
            
            # 利益の標準偏差
            performance["profit_std"] = np.std(performance["profits"]) if performance["profits"] else 0
            
            # シャープレシオ（単純版）
            if performance["profit_std"] > 0:
                performance["sharpe_ratio"] = performance["avg_profit_percent"] / performance["profit_std"]
            else:
                performance["sharpe_ratio"] = 0
    
    return strategy_performance

def visualize_strategy_performance(strategy_performance, output_dir, timestamp):
    """
    戦略別パフォーマンスの可視化
    
    Parameters:
    -----------
    strategy_performance : dict
        戦略別のパフォーマンス指標
    output_dir : str
        出力ディレクトリ
    timestamp : str
        タイムスタンプ
    """
    if not strategy_performance:
        logger.warning("戦略パフォーマンスデータがありません")
        return
    
    try:
        # 戦略別の取引回数と勝率
        plt.figure(figsize=(12, 10))
        
        # 取引回数
        plt.subplot(2, 2, 1)
        strategies = list(strategy_performance.keys())
        counts = [strategy_performance[s]["count"] for s in strategies]
        
        ax = sns.barplot(x=strategies, y=counts)
        for i, count in enumerate(counts):
            ax.text(i, count + 0.5, str(count), ha="center")
        
        plt.title("戦略別取引回数")
        plt.ylabel("取引回数")
        plt.tight_layout()
        
        # 勝率
        plt.subplot(2, 2, 2)
        win_rates = [strategy_performance[s]["win_rate"] for s in strategies]
        
        ax = sns.barplot(x=strategies, y=win_rates)
        for i, rate in enumerate(win_rates):
            ax.text(i, rate + 1, f"{rate:.1f}%", ha="center")
        
        plt.title("戦略別勝率")
        plt.ylabel("勝率 (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        
        # 平均利益率
        plt.subplot(2, 2, 3)
        avg_profits = [strategy_performance[s]["avg_profit_percent"] for s in strategies]
        
        ax = sns.barplot(x=strategies, y=avg_profits)
        for i, profit in enumerate(avg_profits):
            ax.text(i, profit + 0.2, f"{profit:.2f}%", ha="center")
        
        plt.title("戦略別平均利益率")
        plt.ylabel("平均利益率 (%)")
        plt.tight_layout()
        
        # 平均保有期間
        plt.subplot(2, 2, 4)
        avg_durations = [strategy_performance[s]["avg_hold_duration"] for s in strategies]
        
        ax = sns.barplot(x=strategies, y=avg_durations)
        for i, duration in enumerate(avg_durations):
            ax.text(i, duration + 1, f"{duration:.1f}h", ha="center")
        
        plt.title("戦略別平均保有期間")
        plt.ylabel("平均保有期間 (時間)")
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/strategy_performance_{timestamp}.png")
        plt.close()
        
        # 戦略別の利益分布
        if len(strategies) > 0:
            plt.figure(figsize=(12, 6))
            
            for i, strategy in enumerate(strategies):
                if strategy_performance[strategy]["profits"]:
                    plt.subplot(1, len(strategies), i+1)
                    sns.histplot(strategy_performance[strategy]["profits"], kde=True)
                    plt.title(f"{strategy} 利益分布")
                    plt.xlabel("利益率 (%)")
                    plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/profit_distribution_{timestamp}.png")
            plt.close()
        
        # 結果の概要をテキストファイルに保存
        with open(f"{output_dir}/strategy_summary_{timestamp}.txt", "w") as f:
            f.write("戦略別パフォーマンス概要\n")
            f.write("=" * 40 + "\n\n")
            
            for strategy in strategies:
                perf = strategy_performance[strategy]
                f.write(f"戦略: {strategy}\n")
                f.write(f"  取引回数: {perf['count']}\n")
                f.write(f"  勝率: {perf['win_rate']:.2f}%\n")
                f.write(f"  平均利益率: {perf['avg_profit_percent']:.2f}%\n")
                f.write(f"  総利益: {perf['profit']:.4f} USDT\n")
                f.write(f"  平均保有期間: {perf['avg_hold_duration']:.1f}時間\n")
                f.write(f"  利益の標準偏差: {perf['profit_std']:.2f}%\n")
                f.write(f"  シャープレシオ: {perf['sharpe_ratio']:.3f}\n")
                f.write("\n")
    
    except Exception as e:
        logger.error(f"戦略パフォーマンスの可視化中にエラーが発生しました: {e}")
        logger.error(traceback.format_exc())

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Mean Reversion強化版バックテストランナー")
    parser.add_argument("--start", type=str, help="開始時間 (YYYY-MM-DDTHH:MM:SS)")
    parser.add_argument("--end", type=str, help="終了時間 (YYYY-MM-DDTHH:MM:SS)")
    parser.add_argument("--symbol", type=str, help="取引ペア (例: BTCUSDT)")
    parser.add_argument("--interval", type=str, help="時間枠 (例: 1h)")
    parser.add_argument("--output", type=str, default="results", help="出力ディレクトリ")
    args = parser.parse_args()
    
    # 環境変数の読み込み
    load_dotenv()
    
    # バックテスト実行
    logger.info("Mean Reversion強化版バックテストを開始します")
    balance, trades, balance_history = run_backtest(
        start_time=args.start,
        end_time=args.end,
        symbol=args.symbol,
        interval=args.interval,
        output_dir=args.output
    )
    
    # 結果の確認
    if balance is not None:
        logger.info(f"バックテスト完了: 最終残高 {balance:.2f} USDT")
    else:
        logger.warning("バックテストが正常に完了しませんでした。ログを確認してください。")
    
    logger.info("バックテスト完了")

if __name__ == "__main__":
    main()