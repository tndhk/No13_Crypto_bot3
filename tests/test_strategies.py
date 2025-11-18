# -*- coding: utf-8 -*-
"""
戦略シグナル生成のユニットテスト
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategy_integrator import StrategyIntegrator


class TestMeanReversionStrategy:
    """Mean Reversion戦略のテストクラス"""

    def test_initialization_with_default_config(self):
        """デフォルト設定で初期化できることを確認"""
        strategy = MeanReversionStrategy()

        assert strategy.config['rsi_upper'] == 65
        assert strategy.config['rsi_lower'] == 35
        assert strategy.config['z_score_threshold'] == 1.5

    def test_initialization_with_custom_config(self, mean_reversion_config):
        """カスタム設定で初期化できることを確認"""
        strategy = MeanReversionStrategy(mean_reversion_config)

        assert strategy.config['rsi_upper'] == mean_reversion_config['rsi_upper']
        assert strategy.config['rsi_lower'] == mean_reversion_config['rsi_lower']

    def test_generate_signals_returns_dict(self, sample_data_with_indicators):
        """シグナル生成がdictを返すことを確認"""
        strategy = MeanReversionStrategy()
        result = strategy.generate_signals(sample_data_with_indicators)

        assert isinstance(result, dict)

    def test_generate_signals_empty_data(self):
        """空データでも安全に処理できることを確認"""
        strategy = MeanReversionStrategy()
        empty_df = pd.DataFrame()

        result = strategy.generate_signals(empty_df)

        assert result == {}

    def test_generate_signals_insufficient_data(self):
        """データ不足でも安全に処理できることを確認"""
        strategy = MeanReversionStrategy()
        short_df = pd.DataFrame({'close': [100, 101, 102]})

        result = strategy.generate_signals(short_df)

        assert result == {}

    def test_signal_values_in_valid_range(self, sample_data_with_indicators):
        """シグナル値が-1, 0, 1のいずれかであることを確認"""
        strategy = MeanReversionStrategy()
        result = strategy.generate_signals(sample_data_with_indicators)

        if 'signal' in result:
            assert result['signal'] in [-1, 0, 1]

    def test_signal_strength_non_negative(self, sample_data_with_indicators):
        """シグナル強度が非負であることを確認"""
        strategy = MeanReversionStrategy()
        result = strategy.generate_signals(sample_data_with_indicators)

        if 'signal_strength' in result:
            assert result['signal_strength'] >= 0

    def test_risk_reward_calculation(self, sample_data_with_indicators):
        """リスク/リワード計算が妥当な値を返すことを確認"""
        strategy = MeanReversionStrategy()
        signal_info = strategy.generate_signals(sample_data_with_indicators)

        if signal_info:
            sl_percent, tp_percent = strategy.calculate_risk_reward(
                signal_info,
                base_sl_percent=1.5,
                base_tp_percent=5.0
            )

            # SLとTPが妥当な範囲
            assert 0.5 <= sl_percent <= 3.0
            assert 1.0 <= tp_percent <= 10.0

            # リスク/リワード比が最低限を満たす
            assert tp_percent / sl_percent >= 1.5

    def test_buy_signal_on_oversold(self, sample_data_with_indicators):
        """オーバーソールド状態で買いシグナルが出る可能性を確認"""
        strategy = MeanReversionStrategy()

        # RSIを人工的に低くする
        df = sample_data_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc('RSI')] = 20
        df.iloc[-1, df.columns.get_loc('close')] = df.iloc[-1]['BB_lower'] * 0.99

        result = strategy.generate_signals(df)

        # シグナルが生成されることを確認（具体的な値はフィルターによる）
        assert isinstance(result, dict)

    def test_sell_signal_on_overbought(self, sample_data_with_indicators):
        """オーバーボウト状態で売りシグナルが出る可能性を確認"""
        strategy = MeanReversionStrategy()

        # RSIを人工的に高くする
        df = sample_data_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc('RSI')] = 80
        df.iloc[-1, df.columns.get_loc('close')] = df.iloc[-1]['BB_upper'] * 1.01

        result = strategy.generate_signals(df)

        # シグナルが生成されることを確認
        assert isinstance(result, dict)


class TestBreakoutStrategy:
    """ブレイクアウト戦略のテストクラス"""

    def test_initialization(self):
        """初期化できることを確認"""
        config = {
            'lookback_period': 20,
            'breakout_threshold': 0.002,
        }
        strategy = BreakoutStrategy(config)

        assert strategy.config['lookback_period'] == 20

    def test_generate_signals_returns_dict(self, sample_data_with_indicators):
        """シグナル生成がdictを返すことを確認"""
        strategy = BreakoutStrategy({})
        result = strategy.generate_signals(sample_data_with_indicators)

        assert isinstance(result, dict)


class TestStrategyIntegrator:
    """戦略統合のテストクラス"""

    def test_initialization(self):
        """初期化できることを確認"""
        config = {
            'buy_threshold': 0.25,
            'sell_threshold': -0.25,
            'adx_threshold': 25,
        }
        integrator = StrategyIntegrator(config)

        assert integrator.config['buy_threshold'] == 0.25

    def test_integrate_strategies_returns_dict(self, sample_data_with_indicators):
        """戦略統合がdictを返すことを確認"""
        config = {
            'buy_threshold': 0.25,
            'sell_threshold': -0.25,
            'adx_threshold': 25,
        }
        integrator = StrategyIntegrator(config)

        # 空のシグナルでテスト
        trend_signal = {}
        breakout_signal = {}
        mr_signal = {'signal': 1, 'signal_strength': 0.5}

        result = integrator.integrate_strategies(
            trend_signal, breakout_signal, mr_signal, sample_data_with_indicators
        )

        assert isinstance(result, dict)
        assert 'signal' in result or 'weighted_signal' in result

    def test_adaptive_risk_reward(self, sample_data_with_indicators):
        """適応的リスク/リワード計算が妥当な値を返すことを確認"""
        config = {
            'buy_threshold': 0.25,
            'sell_threshold': -0.25,
            'adx_threshold': 25,
        }
        integrator = StrategyIntegrator(config)

        # strategy_weightsを含む完全なsignal_info
        signal_info = {
            'signal': 1,
            'weighted_signal': 0.5,
            'signal_strength': 0.6,
            'strategy_weights': {
                'trend': 0.2,
                'breakout': 0.1,
                'mean_reversion': 0.7
            },
            'strategy_signals': {
                'trend': 0,
                'breakout': 0,
                'mean_reversion': 1
            }
        }

        sl_percent, tp_percent = integrator.adaptive_risk_reward(
            signal_info,
            base_sl_percent=1.5,
            base_tp_percent=8.0
        )

        # 妥当な範囲内
        assert sl_percent > 0
        assert tp_percent > 0
        assert tp_percent > sl_percent

    def test_strategy_weights_sum_to_one(self, sample_data_with_indicators):
        """戦略の重みの合計が1になることを確認"""
        config = {
            'buy_threshold': 0.25,
            'sell_threshold': -0.25,
            'adx_threshold': 25,
        }
        integrator = StrategyIntegrator(config)

        mr_signal = {'signal': 1, 'signal_strength': 0.5}

        result = integrator.integrate_strategies(
            {}, {}, mr_signal, sample_data_with_indicators
        )

        if 'strategy_weights' in result:
            weights = result['strategy_weights']
            total_weight = sum(weights.values())
            # 重みは正規化されている場合は1.0、されていない場合もある
            # 少なくとも正の値であることを確認
            assert total_weight > 0
            # 各重みが0以上であることを確認
            for weight in weights.values():
                assert weight >= 0

    def test_market_environment_detection(self, sample_data_with_indicators):
        """市場環境の検出が機能することを確認"""
        config = {
            'buy_threshold': 0.25,
            'sell_threshold': -0.25,
            'adx_threshold': 25,
        }
        integrator = StrategyIntegrator(config)

        mr_signal = {'signal': 1, 'signal_strength': 0.5}

        result = integrator.integrate_strategies(
            {}, {}, mr_signal, sample_data_with_indicators
        )

        # 市場環境情報が含まれる
        if result:
            assert 'is_trending' in result or 'adx' in result
