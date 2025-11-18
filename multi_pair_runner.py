#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数ペア対応バックテストランナー

複数の取引ペアに対してバックテストを実行し、
ポートフォリオ全体のパフォーマンスを分析します。
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
import pandas as pd

from enhanced_trading_bot import EnhancedTradingBot


class MultiPairRunner:
    """複数ペアのバックテストを管理するクラス"""

    def __init__(self, symbols: List[str] = None):
        """
        マルチペアランナーの初期化

        Parameters:
        -----------
        symbols : list of str, optional
            取引ペアのリスト（例: ['BTCUSDT', 'ETHUSDT']）
            指定しない場合は環境変数から読み込む
        """
        if symbols:
            self.symbols = symbols
        else:
            # 環境変数からカンマ区切りで読み込む
            symbols_str = os.getenv("SYMBOLS", "BTCUSDT")
            self.symbols = [s.strip() for s in symbols_str.split(",")]

        self.results: Dict[str, Dict[str, Any]] = {}
        self.initial_balance = float(os.getenv("INITIAL_BALANCE", "10000"))

        logger.info(f"マルチペアランナー初期化: {len(self.symbols)} ペア")
        logger.info(f"対象ペア: {', '.join(self.symbols)}")

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """
        全ペアのバックテストを実行

        Returns:
        --------
        dict
            各ペアの結果を含む辞書
        """
        for symbol in self.symbols:
            logger.info(f"=== {symbol} のバックテスト開始 ===")

            try:
                result = self._run_single_pair(symbol)
                self.results[symbol] = result

                if result['final_balance']:
                    profit_percent = (result['final_balance'] / self.initial_balance - 1) * 100
                    logger.info(f"{symbol}: 最終残高 {result['final_balance']:.2f} ({profit_percent:+.2f}%)")
                else:
                    logger.warning(f"{symbol}: バックテスト失敗")

            except Exception as e:
                logger.error(f"{symbol} のバックテストでエラー: {e}")
                self.results[symbol] = {
                    'final_balance': None,
                    'trades': [],
                    'error': str(e)
                }

        return self.results

    def _run_single_pair(self, symbol: str) -> Dict[str, Any]:
        """
        単一ペアのバックテストを実行

        Parameters:
        -----------
        symbol : str
            取引ペア

        Returns:
        --------
        dict
            バックテスト結果
        """
        # 環境変数を一時的に更新
        original_symbol = os.getenv("SYMBOL")
        os.environ["SYMBOL"] = symbol

        try:
            bot = EnhancedTradingBot()
            balance, trades = bot.run_backtest()

            return {
                'final_balance': balance,
                'trades': trades if trades else [],
                'trade_count': len([t for t in (trades or []) if t.get('type') == 'BUY']),
                'win_count': len([t for t in (trades or []) if t.get('type') == 'SELL' and t.get('net_profit', 0) > 0]),
                'lose_count': len([t for t in (trades or []) if t.get('type') == 'SELL' and t.get('net_profit', 0) <= 0]),
            }

        finally:
            # 環境変数を復元
            if original_symbol:
                os.environ["SYMBOL"] = original_symbol
            elif "SYMBOL" in os.environ:
                del os.environ["SYMBOL"]

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        ポートフォリオ全体のサマリーを取得

        Returns:
        --------
        dict
            ポートフォリオサマリー
        """
        if not self.results:
            return {}

        total_final_balance = 0
        total_trades = 0
        total_wins = 0
        total_losses = 0
        valid_pairs = 0

        for symbol, result in self.results.items():
            if result.get('final_balance'):
                total_final_balance += result['final_balance']
                total_trades += result.get('trade_count', 0)
                total_wins += result.get('win_count', 0)
                total_losses += result.get('lose_count', 0)
                valid_pairs += 1

        if valid_pairs == 0:
            return {'error': '有効なバックテスト結果がありません'}

        total_initial = self.initial_balance * valid_pairs
        profit = total_final_balance - total_initial
        profit_percent = (total_final_balance / total_initial - 1) * 100
        win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0

        return {
            'pairs': valid_pairs,
            'total_initial_balance': total_initial,
            'total_final_balance': total_final_balance,
            'total_profit': profit,
            'profit_percent': profit_percent,
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'average_profit_per_pair': profit / valid_pairs,
        }

    def save_results(self, output_dir: str = "results") -> str:
        """
        結果をファイルに保存

        Parameters:
        -----------
        output_dir : str
            出力ディレクトリ

        Returns:
        --------
        str
            保存したファイルパス
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # サマリーをJSON形式で保存
        summary = self.get_portfolio_summary()
        summary['symbols'] = self.symbols
        summary['per_pair_results'] = {
            symbol: {
                'final_balance': result.get('final_balance'),
                'trade_count': result.get('trade_count', 0),
                'win_count': result.get('win_count', 0),
                'lose_count': result.get('lose_count', 0),
            }
            for symbol, result in self.results.items()
        }

        filepath = f"{output_dir}/multi_pair_results_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"マルチペア結果を保存: {filepath}")
        return filepath

    def print_summary(self):
        """サマリーをコンソールに出力"""
        summary = self.get_portfolio_summary()

        if 'error' in summary:
            logger.error(summary['error'])
            return

        logger.info("=" * 60)
        logger.info("ポートフォリオサマリー")
        logger.info("=" * 60)
        logger.info(f"取引ペア数: {summary['pairs']}")
        logger.info(f"初期資本合計: {summary['total_initial_balance']:.2f} USDT")
        logger.info(f"最終資本合計: {summary['total_final_balance']:.2f} USDT")
        logger.info(f"総利益: {summary['total_profit']:.2f} USDT ({summary['profit_percent']:.2f}%)")
        logger.info(f"総取引数: {summary['total_trades']}")
        logger.info(f"勝率: {summary['win_rate']:.2f}%")
        logger.info(f"ペアあたり平均利益: {summary['average_profit_per_pair']:.2f} USDT")
        logger.info("=" * 60)

        # ペア別結果
        logger.info("ペア別パフォーマンス:")
        for symbol, result in self.results.items():
            if result.get('final_balance'):
                profit_pct = (result['final_balance'] / self.initial_balance - 1) * 100
                logger.info(f"  {symbol}: {result['final_balance']:.2f} USDT ({profit_pct:+.2f}%)")
            else:
                logger.info(f"  {symbol}: 失敗")


def main():
    """メイン関数"""
    # 環境変数からシンボルリストを読み込んで実行
    runner = MultiPairRunner()
    runner.run_all()
    runner.print_summary()
    runner.save_results()


if __name__ == "__main__":
    main()
