# -*- coding: utf-8 -*-
"""
トレンドフォロー戦略モジュール

移動平均、MACD、RSIなどを組み合わせてトレンドを検出し、シグナルを生成します。
"""

import numpy as np
import pandas as pd

class TrendStrategy:
    def __init__(self, config=None):
        """
        トレンド戦略の初期化
        
        Parameters:
        -----------
        config : dict, optional
            設定パラメータ
        """
        # デフォルト設定
        self.config = {
            'short_window': 9,         # 短期移動平均期間
            'long_window': 21,         # 長期移動平均期間
            'rsi_period': 14,          # RSI期間
            'rsi_oversold': 30,        # RSI買い閾値
            'rsi_overbought': 70,      # RSI売り閾値
            'macd_fast': 12,           # MACD短期
            'macd_slow': 26,           # MACD長期
            'macd_signal': 9,          # MACDシグナル
            'adx_threshold': 25,       # ADX強いトレンド閾値
            'adx_strong_threshold': 35 # ADX非常に強いトレンド閾値
        }
        
        # 設定の上書き
        if config:
            self.config.update(config)
    
    def generate_signals(self, data):
        """
        トレンドフォロー戦略に基づくトレーディングシグナルを生成
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データと指標
            
        Returns:
        --------
        dict
            シグナル情報
        """
        if data.empty or len(data) < self.config['long_window'] + 5:
            return {}
        
        # 最新のデータポイント
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # 基本シグナル
        signal = 0
        signal_strength = 0 # 単純化のため、強度は 1 or 0 とする
        signal_reasons = []
        
        # 1. 移動平均クロスオーバー検出
        if 'EMA_short' in current and 'EMA_long' in current:
            # 短期MAが長期MAを上抜け（買い）
            if (previous['EMA_short'] <= previous['EMA_long'] and 
                current['EMA_short'] > current['EMA_long']):
                signal = 1
                signal_reasons.append(f"EMAクロス(上抜け)")
            
            # 短期MAが長期MAを下抜け（売り）
            elif (previous['EMA_short'] >= previous['EMA_long'] and 
                  current['EMA_short'] < current['EMA_long']):
                signal = -1
                signal_reasons.append(f"EMAクロス(下抜け)")

        # 2. ADXによるトレンド強度の確認 (フィルターとしてのみ使用)
        if signal != 0: # クロスオーバーが発生した場合のみADXをチェック
            adx_value = current.get('ADX', self._calculate_adx(data))
            
            # ADXが閾値未満ならシグナルを無効化
            if adx_value < self.config['adx_threshold']:
                signal = 0 # トレンドがないのでシグナルをキャンセル
                signal_reasons.append(f"ADX閾値未満({adx_value:.1f} < {self.config['adx_threshold']}) - シグナル取消")
            else:
                # トレンドありと判断、シグナル強度をセット
                signal_strength = 1.0 
                signal_reasons.append(f"ADXトレンド確認({adx_value:.1f} >= {self.config['adx_threshold']})")
        
        # シグナル情報をまとめる
        signal_info = {
            'timestamp': current['timestamp'],
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'signal': signal,
            'adx': current.get('ADX', np.nan), # adx_value を使うか、元の値を使うか注意
            'signal_strength': signal_strength, # 1.0 or 0.0
            'signal_reasons': signal_reasons
        }
        
        # 既存の指標情報も追加 (必要最低限)
        for key in ['EMA_short', 'EMA_long', 'ADX']: # RSI, MACD, SMA, ATR は不要に
            if key in current:
                signal_info[key] = current[key]
        
        return signal_info
    
    def _calculate_adx(self, data, period=14):
        """
        ADX（平均方向性指数）の計算
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データ
        period : int
            ADXの期間
            
        Returns:
        --------
        float
            現在のADX値
        """
        if 'ADX' in data.columns:
            return data['ADX'].iloc[-1]
            
        df = data.copy()
        
        # True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # +DM, -DM
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # 14期間の平均
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['tr'].rolling(window=period).mean())
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['tr'].rolling(window=period).mean())
        
        # DX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        
        # ADX
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        return df['adx'].iloc[-1] if not df['adx'].empty else 0