# -*- coding: utf-8 -*-
"""
テスト用フィクスチャの定義
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_ohlcv_data():
    """サンプルOHLCVデータを生成"""
    np.random.seed(42)
    n_periods = 100

    # 基本価格データを生成
    base_price = 50000
    prices = [base_price]

    for _ in range(n_periods - 1):
        change = np.random.normal(0, 0.02) * prices[-1]
        prices.append(prices[-1] + change)

    # DataFrameを作成
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_periods, freq='1h'),
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices],
        'volume': np.random.uniform(100, 1000, n_periods)
    })

    return data


@pytest.fixture
def sample_data_with_indicators(sample_ohlcv_data):
    """インジケーター付きのサンプルデータを生成"""
    df = sample_ohlcv_data.copy()

    # SMA
    df['SMA_short'] = df['close'].rolling(window=3).mean()
    df['SMA_long'] = df['close'].rolling(window=16).mean()

    # EMA
    df['EMA_short'] = df['close'].ewm(span=3, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=16, adjust=False).mean()

    # MA Signal
    df['ma_signal'] = 0
    df.loc[df['EMA_short'] > df['EMA_long'], 'ma_signal'] = 1
    df.loc[df['EMA_short'] < df['EMA_long'], 'ma_signal'] = -1

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2.0 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2.0 * df['BB_std']

    # MACD
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['MACD_hist_prev'] = df['MACD_hist'].shift(1)

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    return df


@pytest.fixture
def mean_reversion_config():
    """Mean Reversion戦略の設定"""
    return {
        'rsi_upper': 65,
        'rsi_lower': 35,
        'rsi_extreme_upper': 75,
        'rsi_extreme_lower': 25,
        'bb_upper_factor': 1.003,
        'bb_lower_factor': 0.997,
        'z_score_threshold': 1.5,
        'ma_deviation': 0.015,
        'price_bounce': 0.002,
        'lookback_period': 20,
    }


@pytest.fixture
def bot_config():
    """ボットの基本設定"""
    return {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'quantity': 0.001,
        'short_window': 3,
        'long_window': 16,
        'stop_loss_percent': 1.5,
        'take_profit_percent': 8.0,
        'maker_fee': 0.001,
        'taker_fee': 0.001,
        'slippage_mean': 0.0005,
        'slippage_std': 0.0003,
    }
