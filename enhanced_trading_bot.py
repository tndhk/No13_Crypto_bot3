# -*- coding: utf-8 -*-
"""
強化されたトレーディングボット - Mean Reversion戦略を最適化

このバージョンでは、Mean Reversion戦略の取引回数と収益性を向上させるために
戦略の検出感度、閾値、および重み付けが最適化されています。
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
import json
import sys
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from config_validator import ConfigValidator, ConfigValidationError
from risk_manager import RiskManager
import warnings
import pickle

# 環境変数の読み込み
load_dotenv()

# ロギング設定
logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
logger.add("logs/enhanced_bot_{time}.log", rotation="1 day", retention="30 days")

# 警告を抑制
warnings.filterwarnings('ignore')

class EnhancedTradingBot:
    def __init__(self):
        """
        強化されたトレーディングボットの初期化
        環境変数から設定を読み込む
        """
        # API設定
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.client = None  # バックテスト時は初期化せず、ライブトレード時のみ初期化
        
        # 取引設定
        self.symbol = os.getenv("SYMBOL", "BTCUSDT")
        self.interval = os.getenv("INTERVAL", "1h")
        self.trade_quantity = float(os.getenv("QUANTITY", "0.001"))
        
        # 基本的な戦略パラメータ
        self.short_window = int(os.getenv("SHORT_WINDOW", "3"))
        self.long_window = int(os.getenv("LONG_WINDOW", "16"))
        self.stop_loss_percent = float(os.getenv("STOP_LOSS_PERCENT", "1.5"))
        self.take_profit_percent = float(os.getenv("TAKE_PROFIT_PERCENT", "8.0"))
        
        # 複合シグナルの重み付け設定
        self.weight_ma = float(os.getenv("WEIGHT_MA", "0.2"))
        self.weight_rsi = float(os.getenv("WEIGHT_RSI", "0.3"))
        self.weight_macd = float(os.getenv("WEIGHT_MACD", "0.2"))
        self.weight_bb = float(os.getenv("WEIGHT_BB", "0.2"))
        self.weight_breakout = float(os.getenv("WEIGHT_BREAKOUT", "0.1"))
        
        # シグナル閾値
        self.buy_threshold = float(os.getenv("BUY_THRESHOLD", "0.15"))
        self.sell_threshold = float(os.getenv("SELL_THRESHOLD", "0.05"))
        
        # フィルター設定
        self.use_complex_signal = os.getenv("USE_COMPLEX_SIGNAL", "true").lower() == "true"
        self.use_price_simulation = os.getenv("USE_PRICE_SIMULATION", "true").lower() == "true"
        self.use_multi_strategy = os.getenv("USE_MULTI_STRATEGY", "true").lower() == "true"

        # 戦略の有効/無効設定
        self.enable_mean_reversion = os.getenv("ENABLE_MEAN_REVERSION", "true").lower() == "true"
        self.enable_trend = os.getenv("ENABLE_TREND", "false").lower() == "true"
        self.enable_breakout = os.getenv("ENABLE_BREAKOUT", "false").lower() == "true"
        
        # コストとスリッページ設定
        self.maker_fee = float(os.getenv("MAKER_FEE", "0.0010"))
        self.taker_fee = float(os.getenv("TAKER_FEE", "0.0010"))
        self.slippage_mean = float(os.getenv("SLIPPAGE_MEAN", "0.0005"))
        self.slippage_std = float(os.getenv("SLIPPAGE_STD", "0.0003"))
        
        # バックテスト設定
        self.execution_delay = int(os.getenv("EXECUTION_DELAY", "1"))
        self.price_simulation_steps = int(os.getenv("PRICE_SIMULATION_STEPS", "100"))
        self.use_cached_data = os.getenv("USE_CACHED_DATA", "true").lower() == "true"
        
        # 状態管理
        self.in_position = False
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.current_trade = {}
        
        # パフォーマンス追跡
        self.trades = []
        self.balance_history = []
        
        # API制限処理
        self.api_request_count = 0
        self.last_api_reset = time.time()
        self.max_requests_per_minute = 1200  # Binanceの制限
        
        # リスク管理設定
        self.MAX_CONSECUTIVE_LOSSES = 3
        self.max_drawdown_limit = 5.0  # パーセント
        self.risk_per_trade = 0.01  # 資本の1%
        self.consecutive_losses = 0
        self.initial_trade_quantity = float(os.getenv("QUANTITY", "0.001"))
        self.trade_quantity = self.initial_trade_quantity
        self.peak_balance = 10000  # バックテスト初期残高
        self.current_balance = 10000
        self.day_start_balance = 10000  # 日次損失管理用
        self.last_trade_date = None

        # 高度なリスク管理の初期化
        risk_config = {
            'max_drawdown_percent': float(os.getenv("MAX_DRAWDOWN_PERCENT", "10.0")),
            'daily_loss_limit_percent': float(os.getenv("DAILY_LOSS_LIMIT_PERCENT", "3.0")),
            'max_consecutive_losses': int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")),
            'trailing_stop_activation': float(os.getenv("TRAILING_STOP_ACTIVATION", "1.5")),
            'trailing_stop_distance': float(os.getenv("TRAILING_STOP_DISTANCE", "0.8")),
            'partial_take_profit_1': float(os.getenv("PARTIAL_TP_1", "1.5")),
            'partial_take_profit_1_ratio': float(os.getenv("PARTIAL_TP_1_RATIO", "0.3")),
            'partial_take_profit_2': float(os.getenv("PARTIAL_TP_2", "3.0")),
            'partial_take_profit_2_ratio': float(os.getenv("PARTIAL_TP_2_RATIO", "0.3")),
            'min_adx_for_trend': float(os.getenv("MIN_ADX_FOR_TREND", "25")),
            'max_adx_for_mean_reversion': float(os.getenv("MAX_ADX_FOR_MR", "20")),
        }
        self.risk_manager = RiskManager(risk_config)
        
        # データディレクトリの確認
        self._ensure_directories()

        # 設定値のバリデーション
        self._validate_configuration()

        # 戦略モジュールのインポート
        self._import_strategies()

        # 戦略オブジェクトの初期化
        self._initialize_strategies()

        # 設定のログ出力
        self._log_configuration()

    def _validate_configuration(self):
        """設定値を検証"""
        try:
            validator = ConfigValidator()
            config = {
                'stop_loss_percent': self.stop_loss_percent,
                'take_profit_percent': self.take_profit_percent,
                'quantity': self.trade_quantity,
                'short_window': self.short_window,
                'long_window': self.long_window,
                'maker_fee': self.maker_fee,
                'taker_fee': self.taker_fee,
                'slippage_mean': self.slippage_mean,
                'slippage_std': self.slippage_std,
                'interval': self.interval,
            }

            warnings = validator.validate(config)

            for warning in warnings:
                logger.warning(f"設定警告: {warning}")

        except ConfigValidationError as e:
            logger.error(f"設定エラー: {e}")
            raise
    
    def _import_strategies(self):
        """戦略モジュールを動的にインポート"""
        global MeanReversionStrategy, TrendStrategy, BreakoutStrategy, StrategyIntegrator
        
        try:
            from strategies import MeanReversionStrategy, TrendStrategy, BreakoutStrategy
            from strategy_integrator import StrategyIntegrator
        except ImportError as e:
            logger.error(f"戦略モジュールのインポートに失敗しました: {e}")
            logger.info("相対インポートを試みます...")
            
            try:
                # 直接インポートを試みる
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                
                from strategies.mean_reversion import MeanReversionStrategy
                from strategies.trend import TrendStrategy
                from strategies.breakout import BreakoutStrategy
                from strategy_integrator import StrategyIntegrator
            except ImportError as e2:
                logger.error(f"戦略モジュールのインポートに再度失敗しました: {e2}")
                raise
    
    def _initialize_strategies(self):
        """戦略オブジェクトを初期化"""
        # 有効な戦略リスト
        self.enabled_strategies = []

        # Mean Reversion戦略の設定
        if self.enable_mean_reversion:
            mr_config = {
                'rsi_upper': float(os.getenv("MR_RSI_UPPER", "65")),
                'rsi_lower': float(os.getenv("MR_RSI_LOWER", "35")),
                'rsi_extreme_upper': float(os.getenv("MR_RSI_EXTREME_UPPER", "75")),
                'rsi_extreme_lower': float(os.getenv("MR_RSI_EXTREME_LOWER", "25")),
                'bb_upper_factor': float(os.getenv("MR_BB_UPPER_FACTOR", "1.003")),
                'bb_lower_factor': float(os.getenv("MR_BB_LOWER_FACTOR", "0.997")),
                'z_score_threshold': float(os.getenv("MR_Z_SCORE_THRESHOLD", "1.5")),
                'ma_deviation': float(os.getenv("MR_MA_DEVIATION", "0.015")),
                'price_bounce': float(os.getenv("MR_PRICE_BOUNCE", "0.002")),
                'lookback_period': 20,
            }
            self.enabled_strategies.append('mean_reversion')
        else:
            mr_config = None

        # トレンド戦略の設定
        if self.enable_trend:
            trend_config = {
                'short_window': self.short_window,
                'long_window': self.long_window,
                'adx_threshold': float(os.getenv("TREND_ADX_THRESHOLD", "25")),
            }
            self.enabled_strategies.append('trend')
        else:
            trend_config = None

        # ブレイクアウト戦略の設定
        if self.enable_breakout:
            breakout_config = {
                'lookback_period': int(os.getenv("BREAKOUT_LOOKBACK", "20")),
                'breakout_threshold': float(os.getenv("BREAKOUT_THRESHOLD", "0.002")),
            }
            self.enabled_strategies.append('breakout')
        else:
            breakout_config = None

        # 戦略統合の設定
        integrator_config = {
            'buy_threshold': float(os.getenv("STRATEGY_BUY_THRESHOLD", "0.25")),
            'sell_threshold': float(os.getenv("STRATEGY_SELL_THRESHOLD", "-0.25")),
            'adx_threshold': float(os.getenv("STRATEGY_ADX_THRESHOLD", "25")),
        }

        # 戦略オブジェクトの作成
        try:
            self.mean_reversion_strategy = MeanReversionStrategy(mr_config) if mr_config else None
            self.trend_strategy = TrendStrategy(trend_config) if trend_config else None
            self.breakout_strategy = BreakoutStrategy(breakout_config) if breakout_config else None
            self.strategy_integrator = StrategyIntegrator(integrator_config)

            if not self.enabled_strategies:
                logger.warning("有効な戦略がありません。少なくとも1つの戦略を有効にしてください。")

        except NameError as e:
            logger.error(f"戦略クラスの初期化に失敗しました: {e}")
            logger.error("strategies モジュールが正しくインポートされているか確認してください")
            raise
    
    def _log_configuration(self):
        """現在の設定をログに出力"""
        logger.info(f"=== 強化ボット設定 ===")
        logger.info(f"取引ペア: {self.symbol}, インターバル: {self.interval}")
        logger.info(f"有効な戦略: {', '.join(self.enabled_strategies) if self.enabled_strategies else 'なし'}")
        logger.info(f"移動平均: 短期={self.short_window}, 長期={self.long_window}")
        logger.info(f"リスク設定: SL={self.stop_loss_percent}%, TP={self.take_profit_percent}%")
        if self.mean_reversion_strategy:
            logger.info(f"平均回帰戦略: RSI閾値={self.mean_reversion_strategy.config['rsi_lower']}/{self.mean_reversion_strategy.config['rsi_upper']}")
        logger.info(f"シグナル閾値: 買い={self.strategy_integrator.config['buy_threshold']}, 売り={self.strategy_integrator.config['sell_threshold']}")
        logger.info(f"価格シミュレーション: {'有効' if self.use_price_simulation else '無効'}")
        logger.info("=" * 30)
    
    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        for directory in ['data', 'logs', 'results', 'cache', 'models']:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_client(self):
        """APIクライアントを初期化（ライブモード用）"""
        if self.client is None:
            try:
                self.client = Client(self.api_key, self.api_secret)
                logger.info("Binance APIクライアント初期化成功")
            except Exception as e:
                logger.error(f"Binance APIクライアント初期化失敗: {e}")
                raise
    
    def _check_api_rate_limit(self):
        """API呼び出し制限をチェックし、必要に応じて待機"""
        current_time = time.time()
        # 1分経過していたらカウンターをリセット
        if current_time - self.last_api_reset > 60:
            self.api_request_count = 0
            self.last_api_reset = current_time
        
        # 制限に近づいたら待機
        if self.api_request_count > self.max_requests_per_minute * 0.9:
            wait_time = 60 - (current_time - self.last_api_reset)
            if wait_time > 0:
                logger.warning(f"API制限に近づいています。{wait_time:.2f}秒待機します")
                time.sleep(wait_time)
                self.api_request_count = 0
                self.last_api_reset = time.time()
        
        self.api_request_count += 1
    
    def calculate_slippage(self, is_buy: bool) -> float:
        """
        スリッページをシミュレート
        
        Parameters:
        -----------
        is_buy : bool
            買い注文かどうか
            
        Returns:
        --------
        float
            スリッページ率（正の値は価格上昇、負の値は価格下落）
        """
        # 正規分布に基づくランダムなスリッページ
        random_slippage = np.random.normal(self.slippage_mean, self.slippage_std)
        
        # 買い注文の場合は正のスリッページ（価格上昇）、売り注文の場合は負のスリッページ（価格下落）
        if is_buy:
            return abs(random_slippage)  # 買いは高くなる（不利）
        else:
            return -abs(random_slippage)  # 売りは安くなる（不利）
    
    def get_historical_data(self, start_time=None, end_time=None, is_backtest=False):
        """
        start_time から end_time までのOHLCVデータをループで取得し、結合して返す
        """
        # キャッシュファイルパス
        cache_file = f"cache/{self.symbol}_{self.interval}_history.pkl"

        # キャッシュを使用する場合はキャッシュをロード
        if self.use_cached_data and is_backtest and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    logger.info(f"キャッシュからデータをロード: {len(data)} ロウソク足")

                    # 日付範囲でフィルタリング
                    if start_time is not None:
                        start_dt = pd.to_datetime(start_time)
                        data = data[data['timestamp'] >= start_dt]

                    if end_time is not None:
                        end_dt = pd.to_datetime(end_time)
                        data = data[data['timestamp'] <= end_dt]

                    if data.empty:
                        logger.warning("キャッシュデータが指定期間内に存在しません")

                    return data
            except (pickle.UnpicklingError, EOFError) as e:
                logger.error(f"キャッシュファイルの読み込みに失敗: {e}")
                logger.info("APIからデータを再取得します")
            except Exception as e:
                logger.error(f"キャッシュ読み込み時の予期しないエラー: {e}")
                logger.info("APIからデータを再取得します")

        # キャッシュが無い場合やライブモードの場合はAPIから取得
        # start_time, end_time を datetime に変換
        start_dt = pd.to_datetime(start_time) if start_time else None
        end_dt = pd.to_datetime(end_time) if end_time else None
        
        logger.info(f"APIからデータをループで取得: {self.symbol}, {self.interval}")

        all_data = []
        current_time = start_dt

        try:
            self._initialize_client()
        except Exception as e:
            logger.error(f"APIクライアントの初期化に失敗: {e}")
            return pd.DataFrame()

        retry_count = 0
        max_retries = 3

        while current_time < end_dt:
            # Binance API は最大1000本のデータを一度に取得可能
            next_time = current_time + pd.Timedelta(minutes=60 * 1000 if self.interval.endswith('h') else 1)
            if next_time > end_dt:
                next_time = end_dt

            try:
                klines = self.client.get_historical_klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    start_str=current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    end_str=next_time.strftime("%Y-%m-%d %H:%M:%S"),
                    limit=1000
                )

                if not klines:
                    break

                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)

                all_data.append(df)
                current_time = df['timestamp'].iloc[-1] + pd.Timedelta(hours=1)  # 次の開始時刻
                retry_count = 0  # 成功したらリセット

            except BinanceAPIException as e:
                logger.error(f"Binance APIエラー: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"最大リトライ回数 ({max_retries}) に達しました")
                    break
                wait_time = 2 ** retry_count
                logger.info(f"{wait_time}秒後にリトライします...")
                time.sleep(wait_time)
                continue
            except Exception as e:
                logger.error(f"データ取得中の予期しないエラー: {e}")
                break

            time.sleep(0.3)  # API制限対策

        if all_data:
            try:
                full_data = pd.concat(all_data).drop_duplicates('timestamp').sort_values('timestamp')

                # キャッシュ保存（オプション）
                if is_backtest:
                    os.makedirs("cache", exist_ok=True)
                    with open(cache_file, 'wb') as f:
                        pickle.dump(full_data, f)
                    logger.info(f"キャッシュ保存完了: {len(full_data)} ロウソク足")

                return full_data
            except Exception as e:
                logger.error(f"データ結合/保存中のエラー: {e}")

        return pd.DataFrame()
    
    def calculate_indicators(self, data):
        """
        テクニカル指標を計算
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データ
            
        Returns:
        --------
        pandas.DataFrame
            指標が追加されたデータフレーム
        """
        if data.empty:
            return data
            
        df = data.copy()
        
        # 移動平均の計算
        # 単純移動平均（SMA）
        df['SMA_short'] = df['close'].rolling(window=self.short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # 指数移動平均（EMA）
        df['EMA_short'] = df['close'].ewm(span=self.short_window, adjust=False).mean()
        df['EMA_long'] = df['close'].ewm(span=self.long_window, adjust=False).mean()
        
        # 移動平均クロスオーバーシグナルの計算
        df['ma_signal'] = 0
        df.loc[df['EMA_short'] > df['EMA_long'], 'ma_signal'] = 1
        df.loc[df['EMA_short'] < df['EMA_long'], 'ma_signal'] = -1
        
        # RSIの計算
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # RSIベースのシグナル
        df['rsi_signal'] = 0
        df.loc[df['RSI'] < 30, 'rsi_signal'] = 1  # 買い
        df.loc[df['RSI'] > 70, 'rsi_signal'] = -1  # 売り
        
        # ボリンジャーバンド
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2.0 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2.0 * df['BB_std']
        
        # ボリンジャーバンドベースのシグナル
        df['bb_signal'] = 0
        df.loc[df['close'] < df['BB_lower'], 'bb_signal'] = 1  # 買い（下限突破）
        df.loc[df['close'] > df['BB_upper'], 'bb_signal'] = -1  # 売り（上限突破）
        
        # MACDの計算
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        df['MACD_hist_prev'] = df['MACD_hist'].shift(1)
        
        # MACDベースのシグナル
        df['macd_signal'] = 0
        df.loc[(df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 'macd_signal'] = 1  # 買い（MACD上抜け）
        df.loc[(df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), 'macd_signal'] = -1  # 売り（MACD下抜け）
        
        # 高値安値ブレイクアウト
        n_periods = 14
        df['highest_high'] = df['high'].rolling(window=n_periods).max()
        df['lowest_low'] = df['low'].rolling(window=n_periods).min()
        
        # ブレイクアウトベースのシグナル
        df['breakout_signal'] = 0
        df.loc[df['close'] > df['highest_high'].shift(1), 'breakout_signal'] = 1  # 買い（高値ブレイク）
        df.loc[df['close'] < df['lowest_low'].shift(1), 'breakout_signal'] = -1  # 売り（安値ブレイク）
        
        # 出来高変化率
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # ATR（Average True Range）- ボラティリティ指標
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # ADX（Average Directional Index）- トレンド強度指標
        # +DM と -DM の計算
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # スムージング
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)

        # DX と ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # ATR比率（ボラティリティの相対指標）
        df['atr_ratio'] = df['ATR'] / df['close']

        return df
    
    def simulate_intracandle_execution(self, candle_data, stop_loss, take_profit):
        """ローソク足内での価格変動を使ってストップロスとテイクプロフィットをシミュレート"""
        # 価格パスが無効な場合はシミュレーションしない
        if not self.use_price_simulation:
            # 単純にローソク足の高値と安値を使って判定
            if candle_data['low'] <= stop_loss:
                return 'Stop Loss', stop_loss
            elif candle_data['high'] >= take_profit:
                return 'Take Profit', take_profit
            return None, None
            
        # ローソク足内での詳細な価格パスをシミュレート
        price_path = self.simulate_detailed_price_path(
            candle_data['open'], 
            candle_data['high'], 
            candle_data['low'], 
            candle_data['close'],
            self.price_simulation_steps
        )
        
        # 価格パス上でのSL/TP発動を検出
        exit_type, exit_price, _ = self.check_sl_tp_on_price_path(price_path, stop_loss, take_profit)
        
        return exit_type, exit_price
    
    def simulate_detailed_price_path(self, open_price, high_price, low_price, close_price, num_steps=None):
        """
        ローソク足内での詳細な価格パスをシミュレーション
        
        Parameters:
        -----------
        open_price, high_price, low_price, close_price : float
            ローソク足のOHLC値
        num_steps : int, optional
            シミュレーションするステップ数（精度）
            
        Returns:
        --------
        list
            時間的に整列された価格系列
        """
        if num_steps is None:
            num_steps = self.price_simulation_steps
            
        # ローソク足の方向性
        is_bullish = close_price > open_price
        
        # 価格変動の振れ幅
        price_range = high_price - low_price
        
        # ブラウン運動に基づく価格パスの生成
        price_path = []
        current_price = open_price
        
        # ランダムウォークに加えて、終値に向かうトレンド成分を追加
        trend_strength = 0.6  # 終値へ引き寄せる力の強さ (0-1)
        
        for i in range(num_steps):
            # 現在のステップの進捗率 (0-1)
            progress = i / (num_steps - 1)
            
            # ランダム成分（ボラティリティ）
            random_component = np.random.normal(0, price_range * 0.03)
            
            # トレンド成分（終値へ向かう力）
            trend_component = (close_price - current_price) * trend_strength * progress
            
            # 価格更新
            current_price += random_component + trend_component
            
            # 高値・安値の範囲内に制約
            current_price = max(min(current_price, high_price * 1.001), low_price * 0.999)
            
            price_path.append(current_price)
        
        # 最後の価格は必ず終値に一致させる
        price_path[-1] = close_price
        
        return price_path

    def check_sl_tp_on_price_path(self, price_path, stop_loss, take_profit):
        """
        価格パス上でのSL/TP発動を検出
        
        Parameters:
        -----------
        price_path : list
            時系列順の価格パス
        stop_loss : float
            ストップロス価格
        take_profit : float
            テイクプロフィット価格
            
        Returns:
        --------
        tuple
            (exit_type, exit_price, exit_index)
            exit_typeは 'Stop Loss', 'Take Profit', None のいずれか
        """
        for i, price in enumerate(price_path):
            if price <= stop_loss:
                return 'Stop Loss', price, i
            if price >= take_profit:
                return 'Take Profit', price, i
        
        return None, None, None
    
    def _calculate_hold_duration(self, start_time, end_time):
        """ポジションの保有期間を計算（時間単位）"""
        if isinstance(start_time, pd.Timestamp) and isinstance(end_time, pd.Timestamp):
            duration = (end_time - start_time).total_seconds() / 3600  # 時間に変換
            return round(duration, 1)
        return 0

    def _calculate_trade_profit(self, entry_price: float, exit_price: float, quantity: float) -> Dict[str, float]:
        """
        取引の利益を計算

        Parameters:
        -----------
        entry_price : float
            エントリー価格
        exit_price : float
            エグジット価格
        quantity : float
            取引数量

        Returns:
        --------
        dict
            gross_profit, net_profit, fees, profit_percent を含む辞書
        """
        gross_profit = (exit_price - entry_price) * quantity
        entry_fee = entry_price * quantity * self.maker_fee
        exit_fee = exit_price * quantity * self.taker_fee
        fees = entry_fee + exit_fee
        net_profit = gross_profit - fees
        profit_percent = (exit_price / entry_price - 1) * 100

        return {
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'fees': fees,
            'profit_percent': profit_percent
        }

    def _process_exit_on_sl_tp(self, current_data: pd.Series, balance: float) -> Tuple[float, Optional[Dict]]:
        """
        ストップロス/テイクプロフィットによるエグジット処理

        Parameters:
        -----------
        current_data : pd.Series
            現在のキャンドルデータ
        balance : float
            現在の残高

        Returns:
        --------
        tuple
            (更新された残高, 取引情報またはNone)
        """
        exit_reason, exit_price = self.simulate_intracandle_execution(
            current_data, self.stop_loss, self.take_profit
        )

        if not exit_reason:
            return balance, None

        # スリッページを適用
        slippage = self.calculate_slippage(is_buy=False)
        execution_price = exit_price * (1 + slippage)

        # 利益計算
        profit_info = self._calculate_trade_profit(
            self.entry_price, execution_price, self.trade_quantity
        )

        # 残高更新
        balance += profit_info['net_profit']

        # 取引記録
        trade = {
            'type': 'SELL',
            'timestamp': current_data['timestamp'],
            'execution_price': execution_price,
            'quantity': self.trade_quantity,
            'gross_profit': profit_info['gross_profit'],
            'net_profit': profit_info['net_profit'],
            'fees': profit_info['fees'],
            'profit_percent': profit_info['profit_percent'],
            'reason': exit_reason,
            'strategy': self.current_trade.get('strategy', 'unknown')
        }

        return balance, trade

    def _process_buy_entry(self, signal_info: Dict, next_candle: pd.Series,
                           current_price: float, current_data: pd.Series, balance: float,
                           initial_balance: float = 10000) -> Optional[Dict]:
        """
        買いエントリー処理

        Parameters:
        -----------
        signal_info : dict
            シグナル情報
        next_candle : pd.Series
            次のキャンドルデータ
        current_price : float
            現在価格
        current_data : pd.Series
            現在のキャンドルデータ
        balance : float
            現在の残高
        initial_balance : float
            初期残高

        Returns:
        --------
        dict or None
            取引情報またはNone
        """
        # リスク管理チェック
        # 1. ドローダウン制限チェック
        can_trade, reason = self.risk_manager.check_drawdown_limit(balance, initial_balance)
        if not can_trade:
            logger.warning(f"エントリー拒否: {reason}")
            return None

        # 2. 日次損失制限チェック
        can_trade, reason = self.risk_manager.check_daily_loss_limit(balance, self.day_start_balance)
        if not can_trade:
            logger.warning(f"エントリー拒否: {reason}")
            return None

        # 3. 連続損失チェック
        can_trade, size_factor = self.risk_manager.check_consecutive_losses()
        if not can_trade:
            logger.warning("エントリー拒否: 連続損失上限到達")
            return None

        # 4. 市場環境フィルター
        # ADXとatr_ratioをsignal_infoに追加
        signal_info_with_market = signal_info.copy()
        signal_info_with_market['adx'] = current_data.get('ADX', 0)
        signal_info_with_market['atr_ratio'] = current_data.get('atr_ratio', 0)

        prev_data = pd.DataFrame([current_data])
        should_skip, skip_reason = self.risk_manager.should_skip_trade(signal_info_with_market, prev_data)
        if should_skip:
            logger.debug(f"エントリースキップ: {skip_reason}")
            return None

        # 5. 高度な市場フィルター（追加データがある場合）
        # Note: This requires more historical data, so we use a simple check here
        should_skip_adv, adv_reason = self.risk_manager.advanced_market_filter(signal_info_with_market, prev_data)
        if should_skip_adv:
            logger.debug(f"高度フィルターによるスキップ: {adv_reason}")
            return None

        # 6. エントリー品質チェック
        entry_timing = self.risk_manager.calculate_optimal_entry_timing(signal_info_with_market, prev_data)
        if entry_timing['entry_quality'] == 'poor':
            # 品質が低い場合、シグナル強度をチェック
            mr_strength = signal_info.get('signal_strengths', {}).get('mean_reversion', 0)
            if mr_strength < 0.7:  # 非常に強いシグナルでなければスキップ
                logger.debug(f"エントリー品質不良: {entry_timing['score']}/5")
                return None

        # ATRベースの動的ポジションサイジング
        atr_value = current_data.get('ATR', None)
        if atr_value and atr_value > 0:
            self.trade_quantity = self.risk_manager.calculate_position_size(balance, atr_value, current_price)
            # 連続損失による追加縮小
            self.trade_quantity *= size_factor
        else:
            # ATRが無い場合はデフォルト値を使用
            risk_amount = balance * self.risk_per_trade
            self.trade_quantity = risk_amount / (current_price * 0.01) * size_factor

        # 実行価格をシミュレート
        execution_price = float(next_candle['open'])
        slippage = self.calculate_slippage(is_buy=True)
        execution_price *= (1 + slippage)

        # リスク/リワード設定（動的計算）
        signal_strength = abs(signal_info.get('weighted_signal', 0.5))
        atr_value = current_data.get('ATR', execution_price * 0.01)

        # ATRベースの動的SL/TP
        if atr_value and atr_value > 0:
            self.stop_loss = self.risk_manager.calculate_dynamic_stop_loss(
                execution_price, atr_value, signal_strength
            )
            self.take_profit = self.risk_manager.calculate_dynamic_take_profit(
                execution_price, atr_value, signal_strength
            )
            sl_percent = (execution_price - self.stop_loss) / execution_price * 100
            tp_percent = (self.take_profit - execution_price) / execution_price * 100
        else:
            # フォールバック: 従来の方法
            sl_percent, tp_percent = self.strategy_integrator.adaptive_risk_reward(
                signal_info, self.stop_loss_percent, self.take_profit_percent
            )
            self.stop_loss = execution_price * (1 - sl_percent/100)
            self.take_profit = execution_price * (1 + tp_percent/100)

        # ポジション設定
        self.entry_price = execution_price
        self.in_position = True

        # リスクマネージャーのポジション状態をリセット
        self.risk_manager.reset_position_state()

        # 主要戦略の特定
        dominant_strategy = max(
            signal_info.get('strategy_weights', {}),
            key=lambda k: signal_info['strategy_weights'][k] if signal_info['strategy_signals'][k] != 0 else 0
        )

        # 取引情報
        trade_info = {
            'type': 'BUY',
            'timestamp': next_candle['timestamp'],
            'signal_timestamp': current_data['timestamp'],
            'signal_price': current_price,
            'execution_price': execution_price,
            'quantity': self.trade_quantity,
            'slippage_percent': (execution_price / float(next_candle['open']) - 1) * 100,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'sl_percent': sl_percent,
            'tp_percent': tp_percent,
            'reason': signal_info.get('signal_reason', f'戦略: {dominant_strategy}'),
            'strategy': dominant_strategy
        }

        self.current_trade = trade_info
        return trade_info

    def _process_sell_exit(self, signal_info: Dict, next_candle: pd.Series,
                           current_price: float, current_data: pd.Series, balance: float) -> Tuple[float, Dict]:
        """
        売りエグジット処理

        Parameters:
        -----------
        signal_info : dict
            シグナル情報
        next_candle : pd.Series
            次のキャンドルデータ
        current_price : float
            現在価格
        current_data : pd.Series
            現在のキャンドルデータ
        balance : float
            現在の残高

        Returns:
        --------
        tuple
            (更新された残高, 取引情報)
        """
        # 実行価格をシミュレート
        execution_price = float(next_candle['open'])
        slippage = self.calculate_slippage(is_buy=False)
        execution_price *= (1 + slippage)

        # 利益計算
        profit_info = self._calculate_trade_profit(
            self.entry_price, execution_price, self.trade_quantity
        )

        # 残高更新
        balance += profit_info['net_profit']

        # 主要戦略の特定
        dominant_strategy = max(
            signal_info.get('strategy_weights', {}),
            key=lambda k: signal_info['strategy_weights'][k] if signal_info['strategy_signals'][k] != 0 else 0
        )

        trade = {
            'type': 'SELL',
            'timestamp': next_candle['timestamp'],
            'signal_timestamp': current_data['timestamp'],
            'signal_price': current_price,
            'execution_price': execution_price,
            'quantity': self.trade_quantity,
            'gross_profit': profit_info['gross_profit'],
            'net_profit': profit_info['net_profit'],
            'fees': profit_info['fees'],
            'profit_percent': profit_info['profit_percent'],
            'reason': signal_info.get('signal_reason', f'戦略: {dominant_strategy}'),
            'strategy': dominant_strategy,
            'entry_price': self.entry_price,
            'hold_duration': self._calculate_hold_duration(
                self.current_trade.get('timestamp', current_data['timestamp']),
                next_candle['timestamp']
            )
        }

        return balance, trade

    def _close_final_position(self, df: pd.DataFrame, balance: float) -> float:
        """
        最終ポジションをクローズ

        Parameters:
        -----------
        df : pd.DataFrame
            データフレーム
        balance : float
            現在の残高

        Returns:
        --------
        float
            更新された残高
        """
        last_price = df.iloc[-1]['close']

        # 利益計算
        profit_info = self._calculate_trade_profit(
            self.entry_price, last_price, self.trade_quantity
        )

        balance += profit_info['net_profit']

        self.trades.append({
            'type': 'SELL',
            'timestamp': df.iloc[-1]['timestamp'],
            'execution_price': last_price,
            'quantity': self.trade_quantity,
            'gross_profit': profit_info['gross_profit'],
            'net_profit': profit_info['net_profit'],
            'fees': profit_info['fees'],
            'profit_percent': profit_info['profit_percent'],
            'reason': 'バックテスト終了',
            'strategy': self.current_trade.get('strategy', 'unknown'),
            'entry_price': self.entry_price,
            'hold_duration': self._calculate_hold_duration(
                self.current_trade.get('timestamp', df.iloc[-1]['timestamp']),
                df.iloc[-1]['timestamp']
            )
        })

        self.in_position = False
        self.current_trade = {}

        return balance
    
    def run_backtest(self):
        """
        強化版：マルチ戦略バックテストを実行
        """
        logger.info("強化版マルチ戦略バックテストモードを開始")
        
        # 環境変数から読み込み
        start_time_str = os.getenv("START_TIME")
        end_time_str = os.getenv("END_TIME")
        
        # 時間変換
        start_time = pd.to_datetime(start_time_str) if start_time_str else None
        end_time = pd.to_datetime(end_time_str) if end_time_str else None
        
        try:
            # データ取得
            data = self.get_historical_data(
                start_time=start_time, 
                end_time=end_time, 
                is_backtest=True
            )
            
            if data.empty:
                logger.error("バックテスト用データが取得できませんでした")
                return None, None
            
            logger.info(f"取得データ: {len(data)} ロウソク足 ({data['timestamp'].min()} - {data['timestamp'].max()})")
            
            # 指標計算
            df = self.calculate_indicators(data)
            
            # 初期資本
            initial_balance = float(os.getenv("INITIAL_BALANCE", "10000"))  # USDT
            balance = initial_balance
            self.in_position = False
            self.trades = []
            self.balance_history = [(df.iloc[0]['timestamp'], balance)]
            self.day_start_balance = initial_balance
            self.last_trade_date = df.iloc[0]['timestamp'].date()

            # リスクマネージャーの初期化
            self.risk_manager.peak_balance = initial_balance
            
            # データポイントが十分かチェック
            min_required_points = max(self.long_window, 26) + 5  # MACDのslow=26が最大値
            if len(df) <= min_required_points:
                logger.error(f"バックテスト用データが不足しています（必要: {min_required_points}, 取得: {len(df)}）")
                return None, None
                
            # 戦略別のシグナルログ
            strategy_signals_log = []
            
            # バックテスト実行
            for i in range(min_required_points, len(df) - 1):
                prev_data = df.iloc[:i+1]
                current_data = df.iloc[i]
                
                # 各戦略のシグナルを計算（有効な戦略のみ）
                trend_signal = self.trend_strategy.generate_signals(prev_data) if self.trend_strategy else {}
                breakout_signal = self.breakout_strategy.generate_signals(prev_data) if self.breakout_strategy else {}
                mean_reversion_signal = self.mean_reversion_strategy.generate_signals(prev_data) if self.mean_reversion_strategy else {}

                # 戦略の統合
                signal_info = self.strategy_integrator.integrate_strategies(
                    trend_signal, breakout_signal, mean_reversion_signal, prev_data
                )
                
                current_price = current_data['close']
                
                # シグナルログ記録
                if signal_info.get('signal', 0) != 0:
                    # シグナル理由を簡潔にまとめる
                    reason_summary = signal_info.get('signal_reason', '')
                    
                    strategy_signals_log.append({
                        'timestamp': signal_info['timestamp'],
                        'price': current_price,
                        'signal': signal_info.get('signal', 0),
                        'weighted_signal': signal_info.get('weighted_signal', 0),
                        'adx': signal_info.get('adx', 0),
                        'is_trending': signal_info.get('is_trending', False),
                        'strategy_signals': signal_info.get('strategy_signals', {}),
                        'strategy_weights': signal_info.get('strategy_weights', {}),
                        'reason': reason_summary
                    })
                
                # ポジションがある場合のSL/TPチェック
                if self.in_position:
                    # トレーリングストップの更新
                    trailing_stop = self.risk_manager.calculate_trailing_stop(
                        self.entry_price, current_price, is_long=True
                    )
                    if trailing_stop and trailing_stop > self.stop_loss:
                        self.stop_loss = trailing_stop

                    # 部分利確のチェック
                    take_qty, remaining_qty = self.risk_manager.check_partial_take_profit(
                        self.entry_price, current_price, self.trade_quantity
                    )
                    if take_qty > 0:
                        # 部分利確を実行
                        partial_profit = (current_price - self.entry_price) * take_qty
                        partial_fee = (self.entry_price * take_qty * self.maker_fee +
                                       current_price * take_qty * self.taker_fee)
                        balance += partial_profit - partial_fee
                        self.trade_quantity = remaining_qty
                        logger.debug(f"部分利確: {take_qty:.6f} @ {current_price:.2f}")

                    # SL/TPチェック
                    balance, trade = self._process_exit_on_sl_tp(current_data, balance)
                    if trade:
                        self.trades.append(trade)
                        self.in_position = False
                        # 取引結果を記録（勝敗判定）
                        is_win = trade.get('net_profit', 0) > 0
                        self.risk_manager.record_trade_result(is_win)
                
                # 次の足でのエントリー
                next_candle_idx = i + self.execution_delay
                if next_candle_idx < len(df):
                    next_candle = df.iloc[next_candle_idx]

                    # 日付が変わったら日次残高をリセット
                    current_date = current_data['timestamp'].date()
                    if current_date != self.last_trade_date:
                        self.day_start_balance = balance
                        self.last_trade_date = current_date

                    # シグナルに基づく取引
                    if signal_info.get('signal', 0) == 1 and not self.in_position:
                        # 買いエントリー処理
                        trade_info = self._process_buy_entry(
                            signal_info, next_candle, current_price, current_data, balance, initial_balance
                        )
                        if trade_info:
                            self.trades.append(trade_info)

                    elif signal_info.get('signal', 0) == -1 and self.in_position:
                        # 売りエグジット処理
                        balance, trade = self._process_sell_exit(
                            signal_info, next_candle, current_price, current_data, balance
                        )
                        self.trades.append(trade)
                        self.in_position = False
                        self.current_trade = {}
                        # 取引結果を記録（勝敗判定）
                        is_win = trade.get('net_profit', 0) > 0
                        self.risk_manager.record_trade_result(is_win)
                
                # 残高履歴を更新
                self.balance_history.append((current_data['timestamp'], balance))
                self.current_balance = balance
                self.peak_balance = max(self.peak_balance, balance)
            
            # 最後のポジションがまだ残っている場合、クローズ
            if self.in_position:
                balance = self._close_final_position(df, balance)
                # 最終残高を更新
                self.balance_history.append((df.iloc[-1]['timestamp'], balance))
            
            # 戦略シグナルの分析
            if strategy_signals_log:
                logger.info(f"マルチ戦略シグナル数: {len(strategy_signals_log)}")
                
                # 戦略別のシグナル回数集計
                strategy_counts = {'trend': 0, 'breakout': 0, 'mean_reversion': 0}
                
                for log in strategy_signals_log:
                    dominant_strategy = max(
                        log.get('strategy_weights', {}), 
                        key=lambda k: log['strategy_weights'][k] if log['strategy_signals'][k] != 0 else 0
                    )
                    if dominant_strategy in strategy_counts:
                        strategy_counts[dominant_strategy] += 1
                
                for strategy, count in strategy_counts.items():
                    logger.info(f"戦略 '{strategy}': {count}回シグナル発生")
            
            # バックテスト結果の分析と出力
            self._analyze_backtest_results(initial_balance, balance)
            
            # 結果の保存
            self._save_backtest_results(initial_balance, balance, df.iloc[0]['timestamp'], df.iloc[-1]['timestamp'])
            
            return balance, self.trades
            
        except Exception as e:
            logger.error(f"バックテストエラー: {e}")
            logger.error(traceback.format_exc())
            return None, None
    
    def _analyze_backtest_results(self, initial_balance, final_balance):
        """バックテスト結果の分析と出力"""
        profit = final_balance - initial_balance
        profit_percent = (final_balance / initial_balance - 1) * 100
        annual_return = profit_percent  # バックテスト期間が1年の場合の近似
        
        # 取引統計
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        
        winning_trades = [t for t in sell_trades if t.get('net_profit', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('net_profit', 0) <= 0]
        
        win_rate = (len(winning_trades) / len(sell_trades) * 100) if sell_trades else 0
        
        # 保有期間の統計
        hold_durations = [t.get('hold_duration', 0) for t in sell_trades if 'hold_duration' in t]
        avg_hold_duration = sum(hold_durations) / len(hold_durations) if hold_durations else 0
        
        # 最大ドローダウンの計算
        balance_df = pd.DataFrame(self.balance_history, columns=['timestamp', 'balance'])
        balance_df['peak'] = balance_df['balance'].cummax()
        balance_df['drawdown'] = (balance_df['peak'] - balance_df['balance']) / balance_df['peak'] * 100
        max_drawdown = balance_df['drawdown'].max()
        
        # リスク調整後リターン指標（シャープレシオ）
        returns = balance_df['balance'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if len(returns) > 1 and returns.std() > 0 else 0
        
        # 結果表示
        logger.info("=" * 60)
        logger.info("強化版マルチ戦略バックテスト結果")
        logger.info("=" * 60)
        logger.info(f"初期資本: {initial_balance:.2f} USDT")
        logger.info(f"最終資本: {final_balance:.2f} USDT")
        logger.info(f"純利益: {profit:.2f} USDT ({profit_percent:.2f}%)")
        logger.info(f"年間収益率: {annual_return:.2f}%")
        logger.info(f"取引数: {len(buy_trades)}")
        logger.info(f"勝率: {win_rate:.2f}%（{len(winning_trades)}勝 / {len(losing_trades)}敗）")
        logger.info(f"最大ドローダウン: {max_drawdown:.2f}%")
        logger.info(f"シャープレシオ: {sharpe_ratio:.3f}")
        logger.info(f"平均保有期間: {avg_hold_duration:.1f}時間")
        
        if winning_trades:
            avg_win = sum(t.get('net_profit', 0) for t in winning_trades) / len(winning_trades)
            max_win = max(t.get('profit_percent', 0) for t in winning_trades)
            logger.info(f"平均利益: {avg_win:.4f} USDT")
            logger.info(f"最大の勝ち: {max_win:.2f}%")
        
        if losing_trades:
            avg_loss = sum(t.get('net_profit', 0) for t in losing_trades) / len(losing_trades)
            max_loss = min(t.get('profit_percent', 0) for t in losing_trades)
            logger.info(f"平均損失: {avg_loss:.4f} USDT")
            logger.info(f"最大の負け: {max_loss:.2f}%")
        
        # プロフィットファクター（総利益 / 総損失）
        total_profit = sum(t.get('net_profit', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('net_profit', 0) for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        logger.info(f"プロフィットファクター: {profit_factor:.2f}")
        
        logger.info("=" * 60)
        
        # 戦略別の分析
        self._analyze_strategy_performance()
    
    def _analyze_strategy_performance(self):
        """戦略別のパフォーマンス分析"""
        strategy_performance = {}
        
        # 取引ごとに戦略を特定し、パフォーマンスを集計
        for trade in self.trades:
            if trade['type'] != 'SELL' or 'strategy' not in trade:
                continue
            
            strategy = trade['strategy']
            
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit': 0,
                    'profit_percent': 0,
                    'hold_durations': []
                }
            
            performance = strategy_performance[strategy]
            performance['count'] += 1
            
            net_profit = trade.get('net_profit', 0)
            profit_percent = trade.get('profit_percent', 0)
            
            if net_profit > 0:
                performance['wins'] += 1
            else:
                performance['losses'] += 1
                
            performance['profit'] += net_profit
            performance['profit_percent'] += profit_percent
            
            if 'hold_duration' in trade:
                performance['hold_durations'].append(trade['hold_duration'])
        
        # 各戦略のパフォーマンス結果を表示
        if strategy_performance:
            logger.info("戦略別パフォーマンス")
            logger.info("-" * 60)
            
            for strategy, performance in strategy_performance.items():
                total_trades = performance['count']
                if total_trades == 0:
                    continue
                    
                win_rate = performance['wins'] / total_trades * 100 if total_trades > 0 else 0
                avg_profit_percent = performance['profit_percent'] / total_trades if total_trades > 0 else 0
                avg_hold_duration = sum(performance['hold_durations']) / len(performance['hold_durations']) if performance['hold_durations'] else 0
                
                logger.info(f"戦略: {strategy}")
                logger.info(f"  取引数: {total_trades}")
                logger.info(f"  勝率: {win_rate:.2f}%")
                logger.info(f"  総利益: {performance['profit']:.4f} USDT")
                logger.info(f"  平均リターン: {avg_profit_percent:.2f}%")
                logger.info(f"  平均保有期間: {avg_hold_duration:.1f}時間")
                logger.info("-" * 60)
    
    def _save_backtest_results(self, initial_balance, final_balance, start_time, end_time):
        """バックテスト結果をファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 結果サマリーをJSON形式で保存
        summary = {
            'start_time': str(start_time),
            'end_time': str(end_time),
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'profit': final_balance - initial_balance,
            'profit_percent': (final_balance / initial_balance - 1) * 100,
            'total_trades': len([t for t in self.trades if t['type'] == 'BUY']),
            'winning_trades': len([t for t in self.trades if t['type'] == 'SELL' and t.get('net_profit', 0) > 0]),
            'losing_trades': len([t for t in self.trades if t['type'] == 'SELL' and t.get('net_profit', 0) <= 0]),
            'timestamp': timestamp
        }
        
        with open(f"results/backtest_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        # 取引詳細をCSV形式で保存
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(f"results/backtest_trades_{timestamp}.csv", index=False)
        
        # 残高履歴をCSV形式で保存
        balance_df = pd.DataFrame(self.balance_history, columns=['timestamp', 'balance'])
        balance_df.to_csv(f"results/backtest_balance_history_{timestamp}.csv", index=False)
        
        logger.info(f"バックテスト結果を保存しました: results/backtest_*_{timestamp}.*")
    
    def run_live_trading(self):
        """
        ライブトレードモードを実行
        
        注意: 実際のお金を使用するため、注意して使用してください
        """
        logger.info("ライブトレードモードは開発中です。バックテストモードをお使いください。")
        # ライブトレード実装は今後の課題

def main():
    """メイン関数"""
    # 引数解析
    parser = argparse.ArgumentParser(description='強化版トレーディングボット')
    parser.add_argument('--mode', choices=['backtest', 'live', 'optimize'], default='backtest',
                        help='実行モード: backtest, live, optimize')
    args = parser.parse_args()
    
    # ボットインスタンス作成
    bot = EnhancedTradingBot()
    
    # 実行モードに応じた処理
    if args.mode == 'backtest':
        bot.run_backtest()
    elif args.mode == 'live':
        bot.run_live_trading()
    elif args.mode == 'optimize':
        logger.info("最適化モードは実装予定です。バックテストモードをお使いください。")

if __name__ == "__main__":
    main()