# -*- coding: utf-8 -*-
"""
インジケーター計算のユニットテスト
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_trading_bot import EnhancedTradingBot


class TestIndicatorCalculation:
    """インジケーター計算のテストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """各テストの前に環境変数を設定"""
        monkeypatch.setenv("USE_CACHED_DATA", "false")
        monkeypatch.setenv("LOG_LEVEL", "ERROR")

    def test_calculate_indicators_returns_dataframe(self, sample_ohlcv_data):
        """インジケーター計算がDataFrameを返すことを確認"""
        bot = EnhancedTradingBot()
        result = bot.calculate_indicators(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)

    def test_sma_calculation(self, sample_ohlcv_data):
        """SMA（単純移動平均）の計算が正しいことを確認"""
        bot = EnhancedTradingBot()
        result = bot.calculate_indicators(sample_ohlcv_data)

        # SMAカラムが存在する
        assert 'SMA_short' in result.columns
        assert 'SMA_long' in result.columns

        # 最初の数行はNaN（ウィンドウサイズ未満）
        assert pd.isna(result['SMA_short'].iloc[0])

        # 手動計算と比較
        short_window = bot.short_window
        expected_sma = sample_ohlcv_data['close'].iloc[:short_window].mean()
        actual_sma = result['SMA_short'].iloc[short_window - 1]

        assert abs(expected_sma - actual_sma) < 0.01

    def test_rsi_calculation(self, sample_ohlcv_data):
        """RSIの計算が0-100の範囲内であることを確認"""
        bot = EnhancedTradingBot()
        result = bot.calculate_indicators(sample_ohlcv_data)

        assert 'RSI' in result.columns

        # NaNを除外して範囲チェック
        rsi_values = result['RSI'].dropna()
        assert all(rsi_values >= 0)
        assert all(rsi_values <= 100)

    def test_bollinger_bands_calculation(self, sample_ohlcv_data):
        """ボリンジャーバンドの計算が正しいことを確認"""
        bot = EnhancedTradingBot()
        result = bot.calculate_indicators(sample_ohlcv_data)

        # 必要なカラムが存在する
        assert 'BB_upper' in result.columns
        assert 'BB_middle' in result.columns
        assert 'BB_lower' in result.columns

        # 上限 > 中央 > 下限の関係を確認（NaNを除外）
        valid_idx = result['BB_upper'].notna()
        assert all(result.loc[valid_idx, 'BB_upper'] >= result.loc[valid_idx, 'BB_middle'])
        assert all(result.loc[valid_idx, 'BB_middle'] >= result.loc[valid_idx, 'BB_lower'])

    def test_macd_calculation(self, sample_ohlcv_data):
        """MACDの計算が正しいことを確認"""
        bot = EnhancedTradingBot()
        result = bot.calculate_indicators(sample_ohlcv_data)

        # 必要なカラムが存在する
        assert 'MACD' in result.columns
        assert 'MACD_signal' in result.columns
        assert 'MACD_hist' in result.columns

        # MACDヒストグラム = MACD - シグナル
        valid_idx = result['MACD_hist'].notna()
        expected_hist = result.loc[valid_idx, 'MACD'] - result.loc[valid_idx, 'MACD_signal']
        actual_hist = result.loc[valid_idx, 'MACD_hist']

        assert np.allclose(expected_hist, actual_hist, rtol=1e-10)

    def test_atr_calculation(self, sample_ohlcv_data):
        """ATRの計算が正の値であることを確認"""
        bot = EnhancedTradingBot()
        result = bot.calculate_indicators(sample_ohlcv_data)

        assert 'ATR' in result.columns

        # ATRは常に正の値
        atr_values = result['ATR'].dropna()
        assert all(atr_values > 0)

    def test_empty_data_handling(self):
        """空のデータフレームを処理できることを確認"""
        bot = EnhancedTradingBot()
        empty_df = pd.DataFrame()

        result = bot.calculate_indicators(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_ma_signal_values(self, sample_ohlcv_data):
        """MAシグナルが-1, 0, 1のいずれかであることを確認"""
        bot = EnhancedTradingBot()
        result = bot.calculate_indicators(sample_ohlcv_data)

        assert 'ma_signal' in result.columns

        # シグナル値が正しい範囲内
        valid_signals = result['ma_signal'].dropna().unique()
        assert all(s in [-1, 0, 1] for s in valid_signals)


class TestSlippageCalculation:
    """スリッページ計算のテストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """各テストの前に環境変数を設定"""
        monkeypatch.setenv("USE_CACHED_DATA", "false")
        monkeypatch.setenv("LOG_LEVEL", "ERROR")

    def test_buy_slippage_positive(self):
        """買い注文のスリッページが正の値であることを確認"""
        bot = EnhancedTradingBot()

        # 複数回テストして傾向を確認
        slippages = [bot.calculate_slippage(is_buy=True) for _ in range(100)]

        # 買いの場合、スリッページは正（価格上昇=不利）
        assert all(s >= 0 for s in slippages)

    def test_sell_slippage_negative(self):
        """売り注文のスリッページが負の値であることを確認"""
        bot = EnhancedTradingBot()

        # 複数回テストして傾向を確認
        slippages = [bot.calculate_slippage(is_buy=False) for _ in range(100)]

        # 売りの場合、スリッページは負（価格下落=不利）
        assert all(s <= 0 for s in slippages)

    def test_slippage_within_reasonable_range(self):
        """スリッページが妥当な範囲内であることを確認"""
        bot = EnhancedTradingBot()

        slippages = [abs(bot.calculate_slippage(is_buy=True)) for _ in range(100)]

        # 平均的なスリッページが設定値に近い
        avg_slippage = np.mean(slippages)
        assert 0.0001 < avg_slippage < 0.002


class TestPriceSimulation:
    """価格シミュレーションのテストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """各テストの前に環境変数を設定"""
        monkeypatch.setenv("USE_CACHED_DATA", "false")
        monkeypatch.setenv("LOG_LEVEL", "ERROR")

    def test_price_path_generation(self):
        """価格パスが正しく生成されることを確認"""
        bot = EnhancedTradingBot()

        open_price = 50000
        high_price = 50500
        low_price = 49500
        close_price = 50200
        num_steps = 50

        price_path = bot.simulate_detailed_price_path(
            open_price, high_price, low_price, close_price, num_steps
        )

        # 正しい長さ
        assert len(price_path) == num_steps

        # 最終価格は終値に一致
        assert price_path[-1] == close_price

        # 価格は高値-安値の範囲に概ね収まる（多少の誤差許容）
        for price in price_path:
            assert price >= low_price * 0.99
            assert price <= high_price * 1.01

    def test_sl_tp_detection_stop_loss(self):
        """ストップロス発動を正しく検出できることを確認"""
        bot = EnhancedTradingBot()

        # 下落する価格パス
        price_path = [50000, 49800, 49500, 49000, 48500]
        stop_loss = 49200
        take_profit = 52000

        exit_type, exit_price, exit_index = bot.check_sl_tp_on_price_path(
            price_path, stop_loss, take_profit
        )

        assert exit_type == 'Stop Loss'
        assert exit_price == 49000
        assert exit_index == 3

    def test_sl_tp_detection_take_profit(self):
        """テイクプロフィット発動を正しく検出できることを確認"""
        bot = EnhancedTradingBot()

        # 上昇する価格パス
        price_path = [50000, 50500, 51000, 51500, 52000]
        stop_loss = 48000
        take_profit = 51200

        exit_type, exit_price, exit_index = bot.check_sl_tp_on_price_path(
            price_path, stop_loss, take_profit
        )

        assert exit_type == 'Take Profit'
        assert exit_price == 51500
        assert exit_index == 3

    def test_sl_tp_no_trigger(self):
        """SL/TPが発動しない場合を正しく処理できることを確認"""
        bot = EnhancedTradingBot()

        # 範囲内の価格パス
        price_path = [50000, 50100, 49900, 50050, 49950]
        stop_loss = 48000
        take_profit = 52000

        exit_type, exit_price, exit_index = bot.check_sl_tp_on_price_path(
            price_path, stop_loss, take_profit
        )

        assert exit_type is None
        assert exit_price is None
        assert exit_index is None
