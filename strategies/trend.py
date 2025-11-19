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
        signal_strength = 0
        signal_reasons = []
        
        # ADXによるトレンド強度の確認
        adx_value = current.get('ADX', self._calculate_adx(data))
        is_trending = adx_value > self.config['adx_threshold']
        
        # トレンドがない場合はシグナルを出さない
        if not is_trending:
            return {
                'timestamp': current['timestamp'],
                'open': current['open'],
                'high': current['high'],
                'low': current['low'],
                'close': current['close'],
                'signal': 0,
                'adx': adx_value,
                'signal_strength': 0,
                'signal_reasons': [f"トレンドなし(ADX={adx_value:.1f})"]
            }
            
        # 1. 移動平均クロスオーバー検出
        ma_signal = 0
        if 'EMA_short' in current and 'EMA_long' in current:
            # 短期MAが長期MAを上抜け（買い）
            if (previous['EMA_short'] <= previous['EMA_long'] and 
                current['EMA_short'] > current['EMA_long']):
                ma_signal = 1
                signal_reasons.append(f"EMAクロス(上抜け)")
            
            # 短期MAが長期MAを下抜け（売り）
            elif (previous['EMA_short'] >= previous['EMA_long'] and 
                  current['EMA_short'] < current['EMA_long']):
                ma_signal = -1
                signal_reasons.append(f"EMAクロス(下抜け)")
            
            # 既存のトレンド継続確認
            elif current['EMA_short'] > current['EMA_long']:
                ma_signal = 0.5 # 上昇トレンド継続
            elif current['EMA_short'] < current['EMA_long']:
                ma_signal = -0.5 # 下降トレンド継続

        # 2. MACD確認
        macd_signal = 0
        if 'MACD' in current and 'MACD_signal' in current:
            if current['MACD'] > current['MACD_signal']:
                macd_signal = 1
            elif current['MACD'] < current['MACD_signal']:
                macd_signal = -1
        
        # 3. シグナル統合
        # クロスオーバーが発生し、かつMACDが同方向の場合に強いシグナル
        if abs(ma_signal) == 1:
            if ma_signal * macd_signal > 0:
                signal = ma_signal
                signal_strength = 0.8
                signal_reasons.append("MACD確認")
                
                # ADXが非常に強い場合はさらに強化
                if adx_value > self.config['adx_strong_threshold']:
                    signal_strength = 1.0
                    signal_reasons.append(f"強いトレンド(ADX={adx_value:.1f})")
            else:
                # MACDと不一致の場合は弱める
                signal = ma_signal
                signal_strength = 0.4
                signal_reasons.append("MACD不一致")
        
        # トレンド継続中の押し目買い/戻り売り（RSI使用）
        elif abs(ma_signal) == 0.5:
            rsi = current.get('RSI', 50)
            
            # 上昇トレンド中の押し目（RSI低下）
            if ma_signal > 0 and rsi < 45 and rsi > 30:
                signal = 1
                signal_strength = 0.6
                signal_reasons.append(f"押し目買い(RSI={rsi:.1f})")
            
            # 下降トレンド中の戻り（RSI上昇）
            elif ma_signal < 0 and rsi > 55 and rsi < 70:
                signal = -1
                signal_strength = 0.6
                signal_reasons.append(f"戻り売り(RSI={rsi:.1f})")

        # シグナル情報をまとめる
        signal_info = {
            'timestamp': current['timestamp'],
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'signal': signal,
            'adx': adx_value,
            'signal_strength': signal_strength,
            'signal_reasons': signal_reasons
        }
        
        # 既存の指標情報も追加
        for key in ['EMA_short', 'EMA_long', 'ADX', 'MACD', 'RSI']:
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