# -*- coding: utf-8 -*-
"""
ブレイクアウト戦略モジュール

価格のレジスタンス/サポートのブレイクアウトを検出してトレードシグナルを生成します。
"""

import numpy as np
import pandas as pd

class BreakoutStrategy:
    def __init__(self, config=None):
        """
        ブレイクアウト戦略の初期化
        
        Parameters:
        -----------
        config : dict, optional
            設定パラメータ
        """
        # デフォルト設定
        self.config = {
            'lookback_period': 20,      # サポート/レジスタンス検出期間
            'breakout_threshold': 0.002, # ブレイクアウト確認閾値 (0.2%)
            'volume_factor': 1.5,        # 出来高の通常比
            'min_range_percent': 0.02,   # 最小レンジ幅（％）
            'vol_filter_threshold': 0.005 # 極端な低ボラティリティ閾値
        }
        
        # 設定の上書き
        if config:
            self.config.update(config)
    
    def generate_signals(self, data):
        """
        ブレイクアウトに基づくトレーディングシグナルを生成
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データと指標
            
        Returns:
        --------
        dict
            シグナル情報
        """
        if data.empty or len(data) < self.config['lookback_period'] + 5:
            return {}
        
        # 最新のデータポイント
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # サポート・レジスタンスを計算
        support, resistance = self._calculate_support_resistance(data)
        
        # 基本シグナル
        signal = 0
        signal_strength = 0
        signal_reasons = []
        
        # ATRの計算（ボラティリティ測定用）
        atr = current.get('ATR', self._calculate_atr(data))
        atr_ratio = atr / current['close']
        
        # ボラティリティが極端に低い場合はシグナルを生成しない
        if atr_ratio < self.config['vol_filter_threshold']:
            return {
                'timestamp': current['timestamp'],
                'open': current['open'],
                'high': current['high'],
                'low': current['low'],
                'close': current['close'],
                'signal': 0,
                'atr_ratio': atr_ratio,
                'signal_strength': 0,
                'resistance': resistance,
                'support': support,
                'breakout_detected': False,
                'signal_reasons': ["ボラティリティ不足"]
            }
        
        # レジスタンスブレイクアウト（買い）
        if resistance is not None:
            # 前回の高値がレジスタンスを下回り、現在の終値がレジスタンスを上回っている場合
            if previous['high'] <= resistance and current['close'] > resistance:
                # 確認：十分な上昇勢い（より確実なブレイクアウト）
                breakout_percent = (current['close'] / resistance - 1)
                if breakout_percent >= self.config['breakout_threshold']:
                    signal = 1
                    signal_strength = min(breakout_percent * 50, 0.8)  # 最大0.8
                    signal_reasons.append(f"レジスタンスブレイク({breakout_percent*100:.2f}%上抜け)")
                    
                    # 出来高増加で確認
                    if 'volume' in current and 'volume_ma' in current:
                        if current['volume'] > current['volume_ma'] * self.config['volume_factor']:
                            signal_strength += 0.2
                            signal_reasons.append("高出来高確認")
        
        # サポートブレイクダウン（売り）
        if support is not None and signal == 0:
            # 前回の安値がサポートを上回り、現在の終値がサポートを下回っている場合
            if previous['low'] >= support and current['close'] < support:
                # 確認：十分な下落勢い
                breakdown_percent = (1 - current['close'] / support)
                if breakdown_percent >= self.config['breakout_threshold']:
                    signal = -1
                    signal_strength = min(breakdown_percent * 50, 0.8)  # 最大0.8
                    signal_reasons.append(f"サポートブレイク({breakdown_percent*100:.2f}%下抜け)")
                    
                    # 出来高増加で確認
                    if 'volume' in current and 'volume_ma' in current:
                        if current['volume'] > current['volume_ma'] * self.config['volume_factor']:
                            signal_strength += 0.2
                            signal_reasons.append("高出来高確認")
        
        # トレンド確認（移動平均の傾きでトレンドを確認）
        if 'SMA_short' in current and signal != 0:
            short_ma_slope = 0
            if len(data) >= 5:
                short_ma_slope = (current['SMA_short'] - data['SMA_short'].iloc[-5]) / data['SMA_short'].iloc[-5]
            
            # 買いシグナルの場合、短期MAも上昇トレンドであれば強化
            if signal > 0 and short_ma_slope > 0.002:
                signal_strength += 0.1
                signal_reasons.append("トレンド一致(上昇)")
            # 売りシグナルの場合、短期MAも下降トレンドであれば強化
            elif signal < 0 and short_ma_slope < -0.002:
                signal_strength += 0.1
                signal_reasons.append("トレンド一致(下降)")
            # トレンドと逆行する場合は弱める
            elif (signal > 0 and short_ma_slope < -0.002) or (signal < 0 and short_ma_slope > 0.002):
                signal_strength *= 0.8
                signal_reasons.append("トレンド逆行")
        
        # シグナル情報をまとめる
        signal_info = {
            'timestamp': current['timestamp'],
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'signal': signal,
            'atr_ratio': atr_ratio,
            'signal_strength': signal_strength,
            'resistance': resistance,
            'support': support,
            'breakout_detected': True if signal != 0 else False,
            'signal_reasons': signal_reasons
        }
        
        # 既存の指標情報も追加
        for key in ['RSI', 'SMA_short', 'SMA_long']:
            if key in current:
                signal_info[key] = current[key]
        
        return signal_info
    
    def _calculate_support_resistance(self, data):
        """
        サポート・レジスタンスレベルの計算（改良版）
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データ
            
        Returns:
        --------
        tuple
            (サポートレベル, レジスタンスレベル)
        """
        if len(data) < self.config['lookback_period']:
            return None, None
        
        lookback = self.config['lookback_period']
        recent_data = data.iloc[-lookback:]
        
        # スイングハイ・ローの検出
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_data) - 2):
            # ローカルトップ（スイングハイ）の検出
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and 
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                swing_highs.append(recent_data['high'].iloc[i])
            
            # ローカルボトム（スイングロー）の検出
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and 
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                swing_lows.append(recent_data['low'].iloc[i])
        
        # レベルのクラスタリング（近いレベルをグループ化）
        def cluster_levels(levels, threshold=0.005):
            if not levels:
                return []
            
            sorted_levels = sorted(levels)
            clusters = [[sorted_levels[0]]]
            
            for level in sorted_levels[1:]:
                latest_cluster = clusters[-1]
                latest_level = latest_cluster[-1]
                
                # 現在のレベルと最新のクラスタの最後のレベルとの差が閾値以内かチェック
                if abs(level - latest_level) / latest_level <= threshold:
                    latest_cluster.append(level)
                else:
                    clusters.append([level])
            
            # 各クラスタの平均値を計算
            return [sum(cluster) / len(cluster) for cluster in clusters]
        
        # クラスタリングされたレベル
        clustered_highs = cluster_levels(swing_highs)
        clustered_lows = cluster_levels(swing_lows)
        
        # 現在の価格
        current_price = data['close'].iloc[-1]
        
        # 有効なレジスタンスとサポートを選択
        resistance = None
        support = None
        
        # レジスタンスは現在価格より上で最も近いもの
        resistance_levels = [r for r in clustered_highs if r > current_price]
        if resistance_levels:
            resistance = min(resistance_levels)
        
        # サポートは現在価格より下で最も近いもの
        support_levels = [s for s in clustered_lows if s < current_price]
        if support_levels:
            support = max(support_levels)
        
        # レンジの幅を確認（狭すぎるレンジは信頼性が低い）
        if resistance is not None and support is not None:
            range_percent = (resistance - support) / support
            if range_percent < self.config['min_range_percent']:
                # レンジが狭すぎる場合は高値/安値を使用
                resistance = recent_data['high'].max()
                support = recent_data['low'].min()
        
        # レジスタンス/サポートが見つからない場合は直近のハイ/ローを使用
        if resistance is None and len(clustered_highs) > 0:
            resistance = max(clustered_highs)
        if support is None and len(clustered_lows) > 0:
            support = min(clustered_lows)
        
        # それでも見つからない場合は期間内の高値/安値
        if resistance is None:
            resistance = recent_data['high'].max()
        if support is None:
            support = recent_data['low'].min()
        
        return support, resistance
    
    def _calculate_atr(self, data, period=14):
        """
        ATR（Average True Range）の計算
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データ
        period : int
            ATRの期間
            
        Returns:
        --------
        float
            現在のATR値
        """
        if 'ATR' in data.columns:
            return data['ATR'].iloc[-1]
            
        df = data.copy()
        
        # True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr.iloc[-1] if not atr.empty else df['close'].iloc[-1] * 0.01  # デフォルト：価格の1%