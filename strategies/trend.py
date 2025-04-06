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
        
        # 1. 移動平均クロスオーバー検出
        if 'EMA_short' in current and 'EMA_long' in current:
            # 短期MAが長期MAを上抜け（買い）
            if (previous['EMA_short'] <= previous['EMA_long'] and 
                current['EMA_short'] > current['EMA_long']):
                signal = 1
                # クロスオーバーの強さを計算（乖離率）
                crossover_strength = (current['EMA_short'] / current['EMA_long'] - 1) * 100
                signal_strength += min(crossover_strength * 5, 0.5)  # 最大0.5
                signal_reasons.append(f"EMAクロス(上抜け:{crossover_strength:.2f}%)")
            
            # 短期MAが長期MAを下抜け（売り）
            elif (previous['EMA_short'] >= previous['EMA_long'] and 
                  current['EMA_short'] < current['EMA_long']):
                signal = -1
                # クロスオーバーの強さを計算
                crossover_strength = (1 - current['EMA_short'] / current['EMA_long']) * 100
                signal_strength += min(crossover_strength * 5, 0.5)  # 最大0.5
                signal_reasons.append(f"EMAクロス(下抜け:{crossover_strength:.2f}%)")
        
        # 2. MACD シグナルクロス
        if 'MACD' in current and 'MACD_signal' in current and signal == 0:
            # MACDがシグナルラインを上抜け（買い）
            if (previous['MACD'] <= previous['MACD_signal'] and 
                current['MACD'] > current['MACD_signal']):
                signal = 1
                # クロスの強さを計算
                macd_strength = abs(current['MACD'] - current['MACD_signal']) / abs(current['MACD_signal']) if current['MACD_signal'] != 0 else 0
                signal_strength += min(macd_strength * 10, 0.6)  # 最大0.6
                signal_reasons.append(f"MACDクロス(上向き)")
                
                # ヒストグラムが加速しているかチェック
                if 'MACD_hist' in current and 'MACD_hist_prev' in current:
                    if current['MACD_hist'] > current['MACD_hist_prev']:
                        signal_strength += 0.1
                        signal_reasons[-1] += "(加速)"
            
            # MACDがシグナルラインを下抜け（売り）
            elif (previous['MACD'] >= previous['MACD_signal'] and 
                  current['MACD'] < current['MACD_signal']):
                signal = -1
                # クロスの強さを計算
                macd_strength = abs(current['MACD'] - current['MACD_signal']) / abs(current['MACD_signal']) if current['MACD_signal'] != 0 else 0
                signal_strength += min(macd_strength * 10, 0.6)  # 最大0.6
                signal_reasons.append(f"MACDクロス(下向き)")
                
                # ヒストグラムが加速しているかチェック
                if 'MACD_hist' in current and 'MACD_hist_prev' in current:
                    if current['MACD_hist'] < current['MACD_hist_prev']:
                        signal_strength += 0.1
                        signal_reasons[-1] += "(加速)"
        
        # 3. RSIによる過剰売買の検出と修正（トレンド継続の確認/フィルタリング）
        if 'RSI' in current and signal != 0:
            # 買いシグナルだがRSIが過買い状態
            if signal > 0 and current['RSI'] > self.config['rsi_overbought']:
                # 過買い状態なのでシグナルを弱める
                signal_strength *= 0.7
                signal_reasons.append(f"RSI過買い警告({current['RSI']:.1f})")
            
            # 売りシグナルだがRSIが過売り状態
            elif signal < 0 and current['RSI'] < self.config['rsi_oversold']:
                # 過売り状態なのでシグナルを弱める
                signal_strength *= 0.7
                signal_reasons.append(f"RSI過売り警告({current['RSI']:.1f})")
            
            # トレンド方向と一致するRSI
            elif (signal > 0 and current['RSI'] > 50) or (signal < 0 and current['RSI'] < 50):
                # トレンドとRSIが一致するのでシグナルを強化
                signal_strength += 0.1
                signal_reasons.append(f"RSIトレンド一致({current['RSI']:.1f})")
        
        # 4. ADXによるトレンド強度の確認
        adx_value = current.get('ADX', self._calculate_adx(data))
        
        # ADXが強いトレンドを示す場合、シグナル強度を加算
        if adx_value > self.config['adx_strong_threshold']:  # 非常に強いトレンド
            signal_strength += 0.2
            signal_reasons.append(f"ADX強トレンド({adx_value:.1f})")
        elif adx_value > self.config['adx_threshold']:  # 強いトレンド
            signal_strength += 0.1
            signal_reasons.append(f"ADXトレンド({adx_value:.1f})")
        elif adx_value < 20 and signal != 0:  # 弱いトレンド - シグナルを弱める
            signal_strength *= 0.8
            signal_reasons.append(f"ADX弱({adx_value:.1f})")
        
        # 5. 価格とMAの位置関係による確認
        if 'SMA_short' in current and 'SMA_long' in current and signal != 0:
            price_vs_ma = 0
            
            # 買いシグナルの場合、価格が両方のMAより上にあればトレンド強化
            if signal > 0:
                if current['close'] > max(current['SMA_short'], current['SMA_long']):
                    price_vs_ma = 1
                    signal_reasons.append("価格>両MA")
                elif current['close'] < min(current['SMA_short'], current['SMA_long']):
                    price_vs_ma = -1
                    signal_reasons.append("価格<両MA")
            
            # 売りシグナルの場合、価格が両方のMAより下にあればトレンド強化
            elif signal < 0:
                if current['close'] < min(current['SMA_short'], current['SMA_long']):
                    price_vs_ma = 1
                    signal_reasons.append("価格<両MA")
                elif current['close'] > max(current['SMA_short'], current['SMA_long']):
                    price_vs_ma = -1
                    signal_reasons.append("価格>両MA")
            
            # シグナル強度を調整
            if price_vs_ma == 1:  # トレンドを強化
                signal_strength += 0.1
            elif price_vs_ma == -1:  # トレンドに反するので強度を下げる
                signal_strength *= 0.8
        
        # 6. ボラティリティフィルター
        if 'ATR' in current:
            atr_ratio = current['ATR'] / current['close']
            
            # 極端に低いボラティリティはトレンドの信頼性を下げる
            if atr_ratio < 0.005 and signal != 0:
                signal_strength *= 0.7
                signal_reasons.append(f"低ボラティリティ({atr_ratio*100:.2f}%)")
            
            # 高ボラティリティは良いが、過度な場合は注意
            elif atr_ratio > 0.02 and signal != 0:
                signal_strength *= 0.8
                signal_reasons.append(f"過度なボラティリティ({atr_ratio*100:.2f}%)")
            elif atr_ratio > 0.01 and signal != 0:
                signal_strength += 0.1
                signal_reasons.append(f"適正ボラティリティ({atr_ratio*100:.2f}%)")
        
        # 最終的なシグナル強度が閾値を超えない場合、シグナルを無効化
        if signal_strength < 0.4 and signal != 0:
            old_signal = signal
            signal = 0
            signal_reasons.append(f"シグナル強度不足({signal_strength:.2f})")
        
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
        for key in ['RSI', 'MACD', 'SMA_short', 'SMA_long', 'ATR']:
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