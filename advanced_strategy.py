# -*- coding: utf-8 -*-
"""
高度な戦略モジュール - 根本的なパフォーマンス改善

問題点:
1. Mean Reversionはトレンド相場では不利
2. BTCは長期的に上昇トレンド
3. 複雑な戦略はBuy and Holdに負けることが多い

解決策:
1. 適応型戦略選択（市場に応じて戦略を切り替え）
2. トレンドフォローをメインに
3. Buy and Holdとの比較で効果を検証
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger


class AdaptiveStrategySelector:
    """市場状況に応じて最適な戦略を自動選択"""

    def __init__(self, config: Dict = None):
        self.config = {
            'trend_adx_threshold': 25,
            'strong_trend_adx': 40,
            'volatility_lookback': 20,
            'regime_lookback': 50,
        }
        if config:
            self.config.update(config)

    def detect_market_regime(self, data: pd.DataFrame) -> Dict:
        """
        市場レジームを検出

        Returns:
        --------
        dict
            regime: 'trending_up', 'trending_down', 'ranging'
            strength: 0-1のスコア
            recommended_strategy: 'trend', 'mean_reversion', 'breakout'
        """
        if len(data) < self.config['regime_lookback']:
            return {
                'regime': 'unknown',
                'strength': 0,
                'recommended_strategy': 'mean_reversion'
            }

        current = data.iloc[-1]
        lookback_data = data.tail(self.config['regime_lookback'])

        # 1. トレンド方向の判定
        close_series = lookback_data['close']
        sma_20 = close_series.rolling(20).mean().iloc[-1]
        sma_50 = close_series.mean()

        # 線形回帰でトレンドの傾きを計算
        x = np.arange(len(close_series))
        slope = np.polyfit(x, close_series.values, 1)[0]
        normalized_slope = slope / close_series.mean() * 100  # パーセントに正規化

        # 2. ADXでトレンド強度を確認
        adx = current.get('ADX', 20)

        # 3. レジーム判定
        if adx > self.config['trend_adx_threshold']:
            if normalized_slope > 0.05:  # 上昇トレンド
                regime = 'trending_up'
                recommended = 'trend'
            elif normalized_slope < -0.05:  # 下降トレンド
                regime = 'trending_down'
                recommended = 'trend'
            else:
                regime = 'ranging'
                recommended = 'mean_reversion'
        else:
            regime = 'ranging'
            recommended = 'mean_reversion'

        # 強度計算
        strength = min(adx / 50, 1.0)

        return {
            'regime': regime,
            'strength': strength,
            'recommended_strategy': recommended,
            'adx': adx,
            'slope': normalized_slope
        }

    def get_strategy_weights(self, regime_info: Dict) -> Dict[str, float]:
        """
        レジームに基づいて戦略の重みを決定

        Returns:
        --------
        dict
            各戦略の重み（合計1.0）
        """
        regime = regime_info['regime']
        strength = regime_info['strength']

        if regime == 'trending_up':
            # 上昇トレンド: トレンドフォローを重視、ショートを避ける
            return {
                'trend': 0.7 * strength + 0.3,
                'breakout': 0.2,
                'mean_reversion': 0.1 * (1 - strength)
            }
        elif regime == 'trending_down':
            # 下降トレンド: 慎重に、ロングを避ける
            return {
                'trend': 0.5 * strength + 0.2,
                'breakout': 0.2,
                'mean_reversion': 0.3 * (1 - strength)
            }
        else:
            # レンジ: Mean Reversionが有効
            return {
                'trend': 0.1,
                'breakout': 0.2,
                'mean_reversion': 0.7
            }


class TrendFollowingStrategy:
    """
    改良版トレンドフォロー戦略

    BTCのような強いトレンド資産に最適化
    """

    def __init__(self, config: Dict = None):
        self.config = {
            'fast_ema': 12,
            'slow_ema': 26,
            'signal_ema': 9,
            'adx_threshold': 20,
            'rsi_overbought': 75,
            'rsi_oversold': 25,
            'atr_multiplier': 2.0,
        }
        if config:
            self.config.update(config)

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """
        トレンドフォローシグナルを生成

        主な特徴:
        1. EMAクロスオーバー
        2. ADXでトレンド強度確認
        3. RSIで過熱感チェック
        4. ATRでボラティリティ調整
        """
        if len(data) < 50:
            return {'signal': 0, 'signal_strength': 0}

        current = data.iloc[-1]
        prev = data.iloc[-2]

        # EMA計算
        close = data['close']
        ema_fast = close.ewm(span=self.config['fast_ema'], adjust=False).mean()
        ema_slow = close.ewm(span=self.config['slow_ema'], adjust=False).mean()

        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]
        prev_fast = ema_fast.iloc[-2]
        prev_slow = ema_slow.iloc[-2]

        # ADX確認
        adx = current.get('ADX', 0)
        plus_di = current.get('plus_di', 0)
        minus_di = current.get('minus_di', 0)

        # RSI確認
        rsi = current.get('RSI', 50)

        # シグナル生成
        signal = 0
        signal_strength = 0
        reasons = []

        # 買いシグナル条件
        buy_conditions = [
            current_fast > current_slow,  # EMAゴールデンクロス
            adx > self.config['adx_threshold'],  # トレンド確認
            plus_di > minus_di,  # 上昇トレンド
            rsi < self.config['rsi_overbought'],  # 過熱していない
        ]

        # 売りシグナル条件（ポジションクローズ用）
        sell_conditions = [
            current_fast < current_slow,  # EMAデッドクロス
            adx > self.config['adx_threshold'],  # トレンド確認
            minus_di > plus_di,  # 下降トレンド
        ]

        # EMAクロスオーバー検出
        golden_cross = prev_fast <= prev_slow and current_fast > current_slow
        death_cross = prev_fast >= prev_slow and current_fast < current_slow

        if golden_cross or (sum(buy_conditions) >= 3 and current_fast > current_slow):
            signal = 1
            signal_strength = min(adx / 50, 1.0)
            reasons.append('EMAゴールデンクロス' if golden_cross else 'EMA上抜け継続')
            if adx > 30:
                reasons.append(f'強いトレンド (ADX={adx:.1f})')

        elif death_cross or sum(sell_conditions) >= 2:
            signal = -1
            signal_strength = min(adx / 50, 1.0)
            reasons.append('EMAデッドクロス' if death_cross else 'EMA下抜け')

        return {
            'signal': signal,
            'signal_strength': signal_strength,
            'signal_reasons': reasons,
            'adx': adx,
            'ema_fast': current_fast,
            'ema_slow': current_slow,
            'rsi': rsi
        }


class MomentumStrategy:
    """
    モメンタム戦略

    価格の勢いを利用した順張り戦略
    """

    def __init__(self, config: Dict = None):
        self.config = {
            'momentum_period': 14,
            'roc_period': 10,
            'strength_threshold': 0.3,
        }
        if config:
            self.config.update(config)

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """
        モメンタムシグナルを生成
        """
        if len(data) < 20:
            return {'signal': 0, 'signal_strength': 0}

        current = data.iloc[-1]
        close = data['close']

        # Rate of Change (ROC)
        roc = (close.iloc[-1] - close.iloc[-self.config['roc_period']]) / close.iloc[-self.config['roc_period']] * 100

        # モメンタム
        momentum = close.iloc[-1] - close.iloc[-self.config['momentum_period']]
        momentum_pct = momentum / close.iloc[-self.config['momentum_period']] * 100

        # RSI
        rsi = current.get('RSI', 50)

        # MACD
        macd_hist = current.get('MACD_hist', 0)

        signal = 0
        signal_strength = 0
        reasons = []

        # 強い上昇モメンタム
        if roc > 2 and momentum_pct > 1 and macd_hist > 0:
            signal = 1
            signal_strength = min(abs(roc) / 10, 1.0)
            reasons.append(f'強い上昇モメンタム (ROC={roc:.1f}%)')

        # 強い下降モメンタム
        elif roc < -2 and momentum_pct < -1 and macd_hist < 0:
            signal = -1
            signal_strength = min(abs(roc) / 10, 1.0)
            reasons.append(f'強い下降モメンタム (ROC={roc:.1f}%)')

        return {
            'signal': signal,
            'signal_strength': signal_strength,
            'signal_reasons': reasons,
            'roc': roc,
            'momentum_pct': momentum_pct
        }


class BuyAndHoldBenchmark:
    """
    Buy and Hold ベンチマーク

    戦略のパフォーマンスを評価するための基準
    """

    @staticmethod
    def calculate_performance(data: pd.DataFrame, initial_balance: float = 10000) -> Dict:
        """
        Buy and Hold のパフォーマンスを計算

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCVデータ
        initial_balance : float
            初期資金

        Returns:
        --------
        dict
            パフォーマンス指標
        """
        if data.empty:
            return {}

        start_price = data.iloc[0]['close']
        end_price = data.iloc[-1]['close']

        # BTC数量（初期購入）
        btc_quantity = initial_balance / start_price

        # 最終価値
        final_value = btc_quantity * end_price

        # リターン
        total_return = (final_value / initial_balance - 1) * 100

        # 最大ドローダウン
        cumulative_max = data['close'].cummax()
        drawdown = (cumulative_max - data['close']) / cumulative_max * 100
        max_drawdown = drawdown.max()

        # ボラティリティ
        daily_returns = data['close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(365) * 100  # 年率換算

        # シャープレシオ（簡易版）
        if volatility > 0:
            sharpe = (total_return / volatility) * np.sqrt(365 / len(data))
        else:
            sharpe = 0

        return {
            'initial_balance': initial_balance,
            'final_balance': final_value,
            'total_return_percent': total_return,
            'max_drawdown_percent': max_drawdown,
            'volatility_annual': volatility,
            'sharpe_ratio': sharpe,
            'start_price': start_price,
            'end_price': end_price,
            'btc_quantity': btc_quantity
        }


class AdvancedStrategyIntegrator:
    """
    高度な戦略統合

    適応型戦略選択 + トレンドフォロー強化
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.adaptive_selector = AdaptiveStrategySelector()
        self.trend_strategy = TrendFollowingStrategy()
        self.momentum_strategy = MomentumStrategy()

    def integrate_signals(self, data: pd.DataFrame,
                         mean_reversion_signal: Dict,
                         breakout_signal: Dict) -> Dict:
        """
        全戦略のシグナルを統合

        Parameters:
        -----------
        data : pd.DataFrame
            市場データ
        mean_reversion_signal : dict
            Mean Reversionシグナル
        breakout_signal : dict
            ブレイクアウトシグナル

        Returns:
        --------
        dict
            統合されたシグナル
        """
        if len(data) < 50:
            return {'signal': 0, 'signal_strength': 0}

        current = data.iloc[-1]

        # 1. 市場レジームを検出
        regime_info = self.adaptive_selector.detect_market_regime(data)

        # 2. 各戦略のシグナルを取得
        trend_signal = self.trend_strategy.generate_signals(data)
        momentum_signal = self.momentum_strategy.generate_signals(data)

        # 3. レジームに基づく重み付け
        weights = self.adaptive_selector.get_strategy_weights(regime_info)

        # 4. シグナルを統合
        signals = {
            'trend': trend_signal.get('signal', 0),
            'momentum': momentum_signal.get('signal', 0),
            'mean_reversion': mean_reversion_signal.get('signal', 0),
            'breakout': breakout_signal.get('signal', 0)
        }

        strengths = {
            'trend': trend_signal.get('signal_strength', 0),
            'momentum': momentum_signal.get('signal_strength', 0),
            'mean_reversion': mean_reversion_signal.get('signal_strength', 0),
            'breakout': breakout_signal.get('signal_strength', 0.5)
        }

        # 加重平均
        weighted_signal = (
            signals['trend'] * weights['trend'] * (1 + strengths['trend']) +
            signals['momentum'] * weights['trend'] * 0.5 +  # モメンタムはトレンドの補助
            signals['mean_reversion'] * weights['mean_reversion'] * (1 + strengths['mean_reversion']) +
            signals['breakout'] * weights['breakout']
        )

        # 最終シグナル決定
        # トレンド相場では閾値を下げて、より積極的にエントリー
        if regime_info['regime'] in ['trending_up', 'trending_down']:
            buy_threshold = 0.15
            sell_threshold = -0.15
        else:
            buy_threshold = 0.25
            sell_threshold = -0.25

        final_signal = 0
        if weighted_signal >= buy_threshold:
            final_signal = 1
        elif weighted_signal <= sell_threshold:
            final_signal = -1

        # 重要: 上昇トレンドではショートを避ける
        if regime_info['regime'] == 'trending_up' and final_signal == -1:
            # ショートシグナルを無視（ポジションクローズのみ許可）
            if signals['trend'] == -1:  # トレンド反転の場合のみ許可
                pass
            else:
                final_signal = 0

        return {
            'timestamp': current['timestamp'],
            'signal': final_signal,
            'weighted_signal': weighted_signal,
            'regime': regime_info,
            'strategy_signals': signals,
            'strategy_weights': weights,
            'strategy_strengths': strengths,
            'trend_details': trend_signal,
            'momentum_details': momentum_signal
        }


def compare_with_benchmark(strategy_result: Dict, benchmark_result: Dict) -> Dict:
    """
    戦略とベンチマーク（Buy and Hold）を比較

    Parameters:
    -----------
    strategy_result : dict
        戦略のパフォーマンス結果
    benchmark_result : dict
        ベンチマークのパフォーマンス結果

    Returns:
    --------
    dict
        比較結果
    """
    strategy_return = strategy_result.get('profit_percent', 0)
    benchmark_return = benchmark_result.get('total_return_percent', 0)

    outperformance = strategy_return - benchmark_return

    return {
        'strategy_return': strategy_return,
        'benchmark_return': benchmark_return,
        'outperformance': outperformance,
        'beat_benchmark': outperformance > 0,
        'strategy_sharpe': strategy_result.get('sharpe_ratio', 0),
        'benchmark_sharpe': benchmark_result.get('sharpe_ratio', 0),
        'strategy_max_dd': strategy_result.get('max_drawdown', 0),
        'benchmark_max_dd': benchmark_result.get('max_drawdown_percent', 0)
    }
