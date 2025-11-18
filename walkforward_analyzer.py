# -*- coding: utf-8 -*-
"""
ウォークフォワード分析モジュール

過学習を防ぎ、戦略の実際のパフォーマンスを評価するための
ウォークフォワード分析を提供します。
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from loguru import logger

from enhanced_trading_bot import EnhancedTradingBot


class WalkForwardAnalyzer:
    """ウォークフォワード分析を実行するクラス"""

    def __init__(self, config: Dict = None):
        """
        ウォークフォワード分析クラスの初期化

        Parameters:
        -----------
        config : dict, optional
            分析設定
        """
        self.config = {
            'in_sample_ratio': 0.7,  # インサンプル期間の比率
            'num_folds': 5,  # 分割数
            'optimization_metric': 'sharpe_ratio',  # 最適化指標
            'min_trades': 10,  # 最小取引数
        }

        if config:
            self.config.update(config)

        self.results: List[Dict] = []

    def run_analysis(self, start_date: str, end_date: str, symbol: str = None) -> Dict:
        """
        ウォークフォワード分析を実行

        Parameters:
        -----------
        start_date : str
            分析開始日（例: '2024-01-01'）
        end_date : str
            分析終了日（例: '2025-01-01'）
        symbol : str, optional
            取引ペア

        Returns:
        --------
        dict
            分析結果
        """
        logger.info(f"ウォークフォワード分析開始: {start_date} - {end_date}")

        # 日付を変換
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_days = (end_dt - start_dt).days

        # 各フォールドの期間を計算
        fold_days = total_days // self.config['num_folds']
        in_sample_days = int(fold_days * self.config['in_sample_ratio'])
        out_sample_days = fold_days - in_sample_days

        all_results = []
        cumulative_balance = float(os.getenv("INITIAL_BALANCE", "10000"))

        for fold in range(self.config['num_folds']):
            logger.info(f"=== フォールド {fold + 1}/{self.config['num_folds']} ===")

            # 期間を計算
            fold_start = start_dt + timedelta(days=fold * fold_days)
            in_sample_end = fold_start + timedelta(days=in_sample_days)
            out_sample_end = in_sample_end + timedelta(days=out_sample_days)

            # インサンプル期間でパラメータ最適化
            logger.info(f"インサンプル期間: {fold_start.date()} - {in_sample_end.date()}")
            optimal_params = self._optimize_parameters(
                fold_start.strftime("%Y-%m-%dT00:00:00"),
                in_sample_end.strftime("%Y-%m-%dT00:00:00"),
                symbol
            )

            # アウトオブサンプル期間でテスト
            logger.info(f"アウトオブサンプル期間: {in_sample_end.date()} - {out_sample_end.date()}")
            oos_result = self._test_parameters(
                optimal_params,
                in_sample_end.strftime("%Y-%m-%dT00:00:00"),
                out_sample_end.strftime("%Y-%m-%dT00:00:00"),
                symbol,
                cumulative_balance
            )

            # 累積残高を更新
            if oos_result.get('final_balance'):
                cumulative_balance = oos_result['final_balance']

            # 結果を保存
            fold_result = {
                'fold': fold + 1,
                'in_sample_period': f"{fold_start.date()} - {in_sample_end.date()}",
                'out_sample_period': f"{in_sample_end.date()} - {out_sample_end.date()}",
                'optimal_params': optimal_params,
                'oos_result': oos_result
            }
            all_results.append(fold_result)

            # フォールド結果をログ
            if oos_result.get('final_balance'):
                profit_pct = (oos_result['final_balance'] / oos_result.get('initial_balance', 10000) - 1) * 100
                logger.info(f"フォールド {fold + 1} OOS結果: {profit_pct:+.2f}%")

        # 全体の結果を集計
        summary = self._calculate_summary(all_results, float(os.getenv("INITIAL_BALANCE", "10000")))

        self.results = all_results

        return {
            'folds': all_results,
            'summary': summary,
            'config': self.config
        }

    def _optimize_parameters(self, start_time: str, end_time: str, symbol: str = None) -> Dict:
        """
        インサンプル期間でパラメータを最適化

        Parameters:
        -----------
        start_time : str
            開始時刻
        end_time : str
            終了時刻
        symbol : str, optional
            取引ペア

        Returns:
        --------
        dict
            最適パラメータ
        """
        # パラメータグリッド（簡略版）
        param_grid = {
            'STOP_LOSS_PERCENT': [1.0, 1.5, 2.0],
            'TAKE_PROFIT_PERCENT': [6.0, 8.0, 10.0],
            'MR_RSI_LOWER': [35, 40],
            'MR_RSI_UPPER': [60, 65],
        }

        best_params = {}
        best_metric = float('-inf')

        # グリッドサーチ
        from itertools import product
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combo in product(*values):
            params = dict(zip(keys, combo))

            # 環境変数を設定
            original_env = {}
            for key, value in params.items():
                original_env[key] = os.getenv(key)
                os.environ[key] = str(value)

            os.environ['START_TIME'] = start_time
            os.environ['END_TIME'] = end_time
            if symbol:
                os.environ['SYMBOL'] = symbol

            try:
                # バックテスト実行
                bot = EnhancedTradingBot()
                balance, trades = bot.run_backtest()

                if balance and trades:
                    # メトリックを計算
                    metric = self._calculate_metric(balance, trades, bot)

                    if metric > best_metric:
                        best_metric = metric
                        best_params = params.copy()

            except Exception as e:
                logger.debug(f"パラメータ最適化エラー: {e}")

            finally:
                # 環境変数を復元
                for key, value in original_env.items():
                    if value is not None:
                        os.environ[key] = value
                    elif key in os.environ:
                        del os.environ[key]

        if not best_params:
            # デフォルトパラメータを返す
            best_params = {
                'STOP_LOSS_PERCENT': 1.5,
                'TAKE_PROFIT_PERCENT': 8.0,
                'MR_RSI_LOWER': 35,
                'MR_RSI_UPPER': 65,
            }

        logger.info(f"最適パラメータ: {best_params}")
        return best_params

    def _test_parameters(self, params: Dict, start_time: str, end_time: str,
                        symbol: str = None, initial_balance: float = 10000) -> Dict:
        """
        パラメータをテスト

        Parameters:
        -----------
        params : dict
            テストするパラメータ
        start_time : str
            開始時刻
        end_time : str
            終了時刻
        symbol : str, optional
            取引ペア
        initial_balance : float
            初期残高

        Returns:
        --------
        dict
            テスト結果
        """
        # 環境変数を設定
        original_env = {}
        for key, value in params.items():
            original_env[key] = os.getenv(key)
            os.environ[key] = str(value)

        os.environ['START_TIME'] = start_time
        os.environ['END_TIME'] = end_time
        os.environ['INITIAL_BALANCE'] = str(initial_balance)
        if symbol:
            os.environ['SYMBOL'] = symbol

        try:
            bot = EnhancedTradingBot()
            balance, trades = bot.run_backtest()

            if balance and trades:
                sell_trades = [t for t in trades if t['type'] == 'SELL']
                win_count = len([t for t in sell_trades if t.get('net_profit', 0) > 0])
                lose_count = len([t for t in sell_trades if t.get('net_profit', 0) <= 0])

                return {
                    'initial_balance': initial_balance,
                    'final_balance': balance,
                    'profit': balance - initial_balance,
                    'profit_percent': (balance / initial_balance - 1) * 100,
                    'trade_count': len([t for t in trades if t['type'] == 'BUY']),
                    'win_rate': win_count / (win_count + lose_count) * 100 if (win_count + lose_count) > 0 else 0,
                    'trades': trades
                }
            else:
                return {
                    'initial_balance': initial_balance,
                    'final_balance': initial_balance,
                    'profit': 0,
                    'profit_percent': 0,
                    'trade_count': 0,
                    'win_rate': 0,
                    'trades': []
                }

        except Exception as e:
            logger.error(f"テストエラー: {e}")
            return {
                'initial_balance': initial_balance,
                'final_balance': initial_balance,
                'error': str(e)
            }

        finally:
            # 環境変数を復元
            for key, value in original_env.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]

    def _calculate_metric(self, balance: float, trades: List, bot: EnhancedTradingBot) -> float:
        """
        最適化メトリックを計算

        Parameters:
        -----------
        balance : float
            最終残高
        trades : list
            取引リスト
        bot : EnhancedTradingBot
            ボットインスタンス

        Returns:
        --------
        float
            メトリック値
        """
        initial_balance = 10000
        sell_trades = [t for t in trades if t['type'] == 'SELL']

        # 取引数が最小値未満の場合はペナルティ
        if len(sell_trades) < self.config['min_trades']:
            return float('-inf')

        metric_name = self.config['optimization_metric']

        if metric_name == 'profit':
            return balance - initial_balance

        elif metric_name == 'sharpe_ratio':
            # シャープレシオを計算
            if hasattr(bot, 'balance_history') and len(bot.balance_history) > 1:
                balance_df = pd.DataFrame(bot.balance_history, columns=['timestamp', 'balance'])
                returns = balance_df['balance'].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    return (returns.mean() / returns.std()) * (252 ** 0.5)
            return 0

        elif metric_name == 'profit_factor':
            # プロフィットファクター
            winning = [t.get('net_profit', 0) for t in sell_trades if t.get('net_profit', 0) > 0]
            losing = [abs(t.get('net_profit', 0)) for t in sell_trades if t.get('net_profit', 0) <= 0]

            total_win = sum(winning) if winning else 0
            total_loss = sum(losing) if losing else 1

            return total_win / total_loss

        elif metric_name == 'win_rate':
            win_count = len([t for t in sell_trades if t.get('net_profit', 0) > 0])
            return win_count / len(sell_trades) * 100 if sell_trades else 0

        else:
            # デフォルト: 利益
            return balance - initial_balance

    def _calculate_summary(self, results: List[Dict], initial_balance: float) -> Dict:
        """
        全体の結果を集計

        Parameters:
        -----------
        results : list
            各フォールドの結果
        initial_balance : float
            初期残高

        Returns:
        --------
        dict
            集計結果
        """
        oos_results = [r['oos_result'] for r in results if r['oos_result'].get('final_balance')]

        if not oos_results:
            return {'error': '有効な結果がありません'}

        # 各フォールドの利益率を計算
        profit_percents = []
        trade_counts = []
        win_rates = []

        for r in oos_results:
            if r.get('profit_percent') is not None:
                profit_percents.append(r['profit_percent'])
            if r.get('trade_count') is not None:
                trade_counts.append(r['trade_count'])
            if r.get('win_rate') is not None:
                win_rates.append(r['win_rate'])

        # 累積リターン
        final_balance = oos_results[-1].get('final_balance', initial_balance) if oos_results else initial_balance
        cumulative_return = (final_balance / initial_balance - 1) * 100

        # 統計
        avg_profit_percent = np.mean(profit_percents) if profit_percents else 0
        std_profit_percent = np.std(profit_percents) if len(profit_percents) > 1 else 0
        avg_trade_count = np.mean(trade_counts) if trade_counts else 0
        avg_win_rate = np.mean(win_rates) if win_rates else 0

        # 一貫性スコア（正の利益のフォールド数）
        positive_folds = len([p for p in profit_percents if p > 0])
        consistency_score = positive_folds / len(profit_percents) * 100 if profit_percents else 0

        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'cumulative_return_percent': cumulative_return,
            'avg_fold_return_percent': avg_profit_percent,
            'std_fold_return_percent': std_profit_percent,
            'avg_trade_count_per_fold': avg_trade_count,
            'avg_win_rate': avg_win_rate,
            'consistency_score': consistency_score,
            'positive_folds': positive_folds,
            'total_folds': len(profit_percents)
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

        filepath = f"{output_dir}/walkforward_analysis_{timestamp}.json"

        # 結果をシリアライズ可能な形式に変換
        serializable_results = []
        for fold in self.results:
            fold_copy = fold.copy()
            if 'oos_result' in fold_copy and 'trades' in fold_copy['oos_result']:
                # 取引データは大きいので要約のみ保存
                fold_copy['oos_result']['trades'] = f"{len(fold_copy['oos_result']['trades'])} trades"
            serializable_results.append(fold_copy)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"ウォークフォワード分析結果を保存: {filepath}")
        return filepath

    def print_summary(self, analysis_result: Dict):
        """
        サマリーをコンソールに出力

        Parameters:
        -----------
        analysis_result : dict
            分析結果
        """
        summary = analysis_result.get('summary', {})

        if 'error' in summary:
            logger.error(summary['error'])
            return

        logger.info("=" * 60)
        logger.info("ウォークフォワード分析サマリー")
        logger.info("=" * 60)
        logger.info(f"分割数: {summary.get('total_folds', 0)}")
        logger.info(f"初期残高: {summary.get('initial_balance', 0):.2f} USDT")
        logger.info(f"最終残高: {summary.get('final_balance', 0):.2f} USDT")
        logger.info(f"累積リターン: {summary.get('cumulative_return_percent', 0):.2f}%")
        logger.info(f"平均フォールドリターン: {summary.get('avg_fold_return_percent', 0):.2f}% (±{summary.get('std_fold_return_percent', 0):.2f}%)")
        logger.info(f"平均取引数/フォールド: {summary.get('avg_trade_count_per_fold', 0):.1f}")
        logger.info(f"平均勝率: {summary.get('avg_win_rate', 0):.2f}%")
        logger.info(f"一貫性スコア: {summary.get('consistency_score', 0):.0f}% ({summary.get('positive_folds', 0)}/{summary.get('total_folds', 0)}フォールドが正)")
        logger.info("=" * 60)


def main():
    """メイン関数"""
    # 分析期間を設定
    start_date = os.getenv("START_TIME", "2024-01-01")[:10]
    end_date = os.getenv("END_TIME", "2025-01-01")[:10]

    # ウォークフォワード分析を実行
    analyzer = WalkForwardAnalyzer({
        'num_folds': 4,
        'in_sample_ratio': 0.7,
        'optimization_metric': 'sharpe_ratio'
    })

    result = analyzer.run_analysis(start_date, end_date)
    analyzer.print_summary(result)
    analyzer.save_results()


if __name__ == "__main__":
    main()
