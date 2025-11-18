#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バックテスト結果の可視化モジュール

バックテスト結果をグラフ化し、パフォーマンスを視覚的に分析します。
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

# 日本語フォント対応（利用可能な場合）
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
except Exception:
    pass

plt.rcParams['figure.figsize'] = (14, 8)
plt.style.use('seaborn-v0_8-whitegrid')


class BacktestVisualizer:
    """バックテスト結果を可視化するクラス"""

    def __init__(self, results_dir: str = "results"):
        """
        可視化クラスの初期化

        Parameters:
        -----------
        results_dir : str
            結果ファイルが保存されているディレクトリ
        """
        self.results_dir = results_dir

    def plot_balance_history(self, balance_history: List[tuple],
                              trades: List[Dict] = None,
                              title: str = "Balance History",
                              save_path: str = None) -> None:
        """
        残高履歴をプロット

        Parameters:
        -----------
        balance_history : list of tuple
            (timestamp, balance) のリスト
        trades : list of dict, optional
            取引リスト（エントリー/エグジットポイントを表示）
        title : str
            グラフタイトル
        save_path : str, optional
            保存先パス
        """
        if not balance_history:
            logger.warning("残高履歴データがありません")
            return

        df = pd.DataFrame(balance_history, columns=['timestamp', 'balance'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig, ax = plt.subplots(figsize=(14, 6))

        # 残高曲線
        ax.plot(df['timestamp'], df['balance'], 'b-', linewidth=1.5, label='Balance')

        # 初期残高ライン
        initial_balance = df['balance'].iloc[0]
        ax.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.5, label='Initial Balance')

        # 取引ポイントをプロット
        if trades:
            buy_trades = [t for t in trades if t.get('type') == 'BUY']
            sell_trades = [t for t in trades if t.get('type') == 'SELL']

            # 買いポイント
            for trade in buy_trades:
                ts = pd.to_datetime(trade.get('timestamp'))
                idx = df[df['timestamp'] >= ts].index
                if len(idx) > 0:
                    balance_at_time = df.loc[idx[0], 'balance']
                    ax.scatter(ts, balance_at_time, color='green', marker='^', s=50, alpha=0.7)

            # 売りポイント（利益/損失で色分け）
            for trade in sell_trades:
                ts = pd.to_datetime(trade.get('timestamp'))
                idx = df[df['timestamp'] >= ts].index
                if len(idx) > 0:
                    balance_at_time = df.loc[idx[0], 'balance']
                    color = 'blue' if trade.get('net_profit', 0) > 0 else 'red'
                    ax.scatter(ts, balance_at_time, color=color, marker='v', s=50, alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Balance (USDT)')
        ax.set_title(title)
        ax.legend()

        # 日付フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"残高履歴グラフを保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_drawdown(self, balance_history: List[tuple],
                       title: str = "Drawdown Analysis",
                       save_path: str = None) -> None:
        """
        ドローダウンをプロット

        Parameters:
        -----------
        balance_history : list of tuple
            (timestamp, balance) のリスト
        title : str
            グラフタイトル
        save_path : str, optional
            保存先パス
        """
        if not balance_history:
            logger.warning("残高履歴データがありません")
            return

        df = pd.DataFrame(balance_history, columns=['timestamp', 'balance'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['peak'] = df['balance'].cummax()
        df['drawdown'] = (df['peak'] - df['balance']) / df['peak'] * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # 残高と高値
        ax1.plot(df['timestamp'], df['balance'], 'b-', label='Balance')
        ax1.plot(df['timestamp'], df['peak'], 'g--', alpha=0.5, label='Peak')
        ax1.set_ylabel('Balance (USDT)')
        ax1.legend()
        ax1.set_title('Balance and Peak')

        # ドローダウン
        ax2.fill_between(df['timestamp'], df['drawdown'], color='red', alpha=0.3)
        ax2.plot(df['timestamp'], df['drawdown'], 'r-', linewidth=1)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdown')
        ax2.invert_yaxis()  # ドローダウンは下方向

        # 最大ドローダウンをマーク
        max_dd_idx = df['drawdown'].idxmax()
        max_dd = df.loc[max_dd_idx, 'drawdown']
        max_dd_date = df.loc[max_dd_idx, 'timestamp']
        ax2.scatter(max_dd_date, max_dd, color='darkred', s=100, zorder=5)
        ax2.annotate(f'Max: {max_dd:.2f}%', (max_dd_date, max_dd),
                    xytext=(10, 10), textcoords='offset points')

        # 日付フォーマット
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"ドローダウングラフを保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_trade_analysis(self, trades: List[Dict],
                            title: str = "Trade Analysis",
                            save_path: str = None) -> None:
        """
        取引分析をプロット

        Parameters:
        -----------
        trades : list of dict
            取引リスト
        title : str
            グラフタイトル
        save_path : str, optional
            保存先パス
        """
        if not trades:
            logger.warning("取引データがありません")
            return

        # 売り取引（決済）のみを対象
        sell_trades = [t for t in trades if t.get('type') == 'SELL']

        if not sell_trades:
            logger.warning("決済取引がありません")
            return

        profits = [t.get('profit_percent', 0) for t in sell_trades]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 利益/損失の分布
        ax1 = axes[0, 0]
        colors = ['green' if p > 0 else 'red' for p in profits]
        ax1.bar(range(len(profits)), profits, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Trade #')
        ax1.set_ylabel('Profit (%)')
        ax1.set_title('Profit/Loss per Trade')

        # 2. 利益の累積
        ax2 = axes[0, 1]
        cumulative = np.cumsum(profits)
        ax2.plot(cumulative, 'b-', linewidth=1.5)
        ax2.fill_between(range(len(cumulative)), cumulative, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Trade #')
        ax2.set_ylabel('Cumulative Profit (%)')
        ax2.set_title('Cumulative Profit')

        # 3. 利益の分布（ヒストグラム）
        ax3 = axes[1, 0]
        ax3.hist(profits, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax3.axvline(x=np.mean(profits), color='red', linestyle='--', label=f'Mean: {np.mean(profits):.2f}%')
        ax3.set_xlabel('Profit (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Profit Distribution')
        ax3.legend()

        # 4. 勝敗比率（パイチャート）
        ax4 = axes[1, 1]
        wins = sum(1 for p in profits if p > 0)
        losses = sum(1 for p in profits if p <= 0)
        sizes = [wins, losses]
        labels = [f'Wins ({wins})', f'Losses ({losses})']
        colors_pie = ['green', 'red']
        ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Win/Loss Ratio')

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"取引分析グラフを保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_strategy_performance(self, trades: List[Dict],
                                   title: str = "Strategy Performance",
                                   save_path: str = None) -> None:
        """
        戦略別パフォーマンスをプロット

        Parameters:
        -----------
        trades : list of dict
            取引リスト
        title : str
            グラフタイトル
        save_path : str, optional
            保存先パス
        """
        if not trades:
            logger.warning("取引データがありません")
            return

        # 戦略別に集計
        sell_trades = [t for t in trades if t.get('type') == 'SELL']
        strategy_data = {}

        for trade in sell_trades:
            strategy = trade.get('strategy', 'unknown')
            if strategy not in strategy_data:
                strategy_data[strategy] = {'profits': [], 'count': 0}
            strategy_data[strategy]['profits'].append(trade.get('profit_percent', 0))
            strategy_data[strategy]['count'] += 1

        if not strategy_data:
            logger.warning("戦略データがありません")
            return

        strategies = list(strategy_data.keys())
        avg_profits = [np.mean(strategy_data[s]['profits']) for s in strategies]
        trade_counts = [strategy_data[s]['count'] for s in strategies]
        win_rates = [sum(1 for p in strategy_data[s]['profits'] if p > 0) / len(strategy_data[s]['profits']) * 100
                     for s in strategies]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 平均利益
        colors = ['green' if p > 0 else 'red' for p in avg_profits]
        axes[0].bar(strategies, avg_profits, color=colors, alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_ylabel('Average Profit (%)')
        axes[0].set_title('Average Profit by Strategy')
        axes[0].tick_params(axis='x', rotation=45)

        # 取引回数
        axes[1].bar(strategies, trade_counts, color='steelblue', alpha=0.7)
        axes[1].set_ylabel('Number of Trades')
        axes[1].set_title('Trade Count by Strategy')
        axes[1].tick_params(axis='x', rotation=45)

        # 勝率
        axes[2].bar(strategies, win_rates, color='orange', alpha=0.7)
        axes[2].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('Win Rate (%)')
        axes[2].set_title('Win Rate by Strategy')
        axes[2].tick_params(axis='x', rotation=45)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"戦略パフォーマンスグラフを保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_report(self, balance_history: List[tuple],
                        trades: List[Dict],
                        output_dir: str = None) -> str:
        """
        総合レポートを生成

        Parameters:
        -----------
        balance_history : list of tuple
            残高履歴
        trades : list of dict
            取引リスト
        output_dir : str, optional
            出力ディレクトリ

        Returns:
        --------
        str
            レポートディレクトリパス
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{self.results_dir}/report_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)

        # 各グラフを生成
        self.plot_balance_history(
            balance_history, trades,
            save_path=f"{output_dir}/balance_history.png"
        )

        self.plot_drawdown(
            balance_history,
            save_path=f"{output_dir}/drawdown.png"
        )

        self.plot_trade_analysis(
            trades,
            save_path=f"{output_dir}/trade_analysis.png"
        )

        self.plot_strategy_performance(
            trades,
            save_path=f"{output_dir}/strategy_performance.png"
        )

        logger.info(f"レポートを生成しました: {output_dir}")
        return output_dir


def main():
    """テスト用メイン関数"""
    # サンプルデータでテスト
    import numpy as np
    from datetime import timedelta

    # サンプル残高履歴
    base_time = datetime(2024, 1, 1)
    balance = 10000
    balance_history = []

    for i in range(100):
        timestamp = base_time + timedelta(hours=i)
        balance += np.random.normal(10, 50)
        balance_history.append((timestamp, max(balance, 8000)))

    # サンプル取引
    trades = []
    for i in range(20):
        trades.append({
            'type': 'BUY',
            'timestamp': base_time + timedelta(hours=i*5),
            'strategy': np.random.choice(['mean_reversion', 'trend', 'breakout'])
        })
        trades.append({
            'type': 'SELL',
            'timestamp': base_time + timedelta(hours=i*5+3),
            'net_profit': np.random.normal(20, 50),
            'profit_percent': np.random.normal(1, 3),
            'strategy': trades[-1]['strategy']
        })

    # 可視化
    visualizer = BacktestVisualizer()
    visualizer.generate_report(balance_history, trades)


if __name__ == "__main__":
    main()
