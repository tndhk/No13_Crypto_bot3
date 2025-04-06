# -*- coding: utf-8 -*-
"""
強化されたMean Reversion戦略モジュール

取引回数を増やし、パフォーマンスを向上させるために最適化されています。
"""

import numpy as np
import pandas as pd

class MeanReversionStrategy:
    def __init__(self, config=None):
        """
        Mean Reversion戦略の初期化
        
        Parameters:
        -----------
        config : dict, optional
            設定パラメータ（デフォルト値を上書き）
        """
        # デフォルト設定
        self.config = {
            'rsi_upper': 65,           # RSI上限閾値（旧：70）
            'rsi_lower': 35,           # RSI下限閾値（旧：30）
            'rsi_extreme_upper': 75,   # RSI極端上限
            'rsi_extreme_lower': 25,   # RSI極端下限
            'bb_upper_factor': 1.003,  # ボリンジャーバンド上限乗数（旧：1.005）
            'bb_lower_factor': 0.997,  # ボリンジャーバンド下限乗数（旧：0.995）
            'z_score_threshold': 1.5,  # Z-スコア閾値（旧：2.0）
            'ma_deviation': 0.015,     # 移動平均からの乖離閾値（新規）
            'price_bounce': 0.002,     # 価格反発率閾値（新規）
            'vol_high_threshold': 0.018,  # 高ボラティリティ閾値（旧：0.015）
            'vol_low_threshold': 0.004,   # 低ボラティリティ閾値（新規）
            'lookback_period': 20,        # Z-スコア計算期間（旧：30）
        }
        
        # 設定の上書き
        if config:
            self.config.update(config)
    
    def generate_signals(self, data):
        """
        強化版：平均回帰（Mean Reversion）に基づくトレーディングシグナルを生成
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データと指標
            
        Returns:
        --------
        dict
            シグナル情報
        """
        if data.empty or len(data) < 30:
            return {}
        
        # 最新のデータポイント
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # 基本シグナル
        signal = 0
        signal_strength = 0
        signal_reasons = []
        
        # 1. ボリンジャーバンドを利用した平均回帰
        if 'BB_upper' in current and 'BB_middle' in current and 'BB_lower' in current:
            # 買いシグナル
            if current['close'] < current['BB_lower'] * self.config['bb_lower_factor']:
                if current['RSI'] < self.config['rsi_lower']:
                    signal = 1
                    signal_strength += (self.config['rsi_lower'] - current['RSI']) / self.config['rsi_lower']
                    signal_reasons.append(f"BB下限突破(BB:{current['BB_lower']:.2f}×{self.config['bb_lower_factor']:.3f}, 価格:{current['close']:.2f})")
            
            # 売りシグナル
            elif current['close'] > current['BB_upper'] * self.config['bb_upper_factor']:
                if current['RSI'] > self.config['rsi_upper']:
                    signal = -1
                    signal_strength += (current['RSI'] - self.config['rsi_upper']) / (100 - self.config['rsi_upper'])
                    signal_reasons.append(f"BB上限突破(BB:{current['BB_upper']:.2f}×{self.config['bb_upper_factor']:.3f}, 価格:{current['close']:.2f})")
        
        # 2. ボリンジャーバンドからの反発を検出
        if signal == 0 and len(data) > 2:
            # 買いシグナル - 下限からの反発
            if previous['close'] < previous['BB_lower'] and current['close'] > previous['close']:
                bounce_percent = (current['close'] - previous['close']) / previous['close']
                if bounce_percent > self.config['price_bounce']:  # 一定以上の反発
                    signal = 1
                    signal_strength += 0.5
                    signal_reasons.append(f"BB下限からの反発({bounce_percent*100:.2f}%)")
            
            # 売りシグナル - 上限からの反発
            elif previous['close'] > previous['BB_upper'] and current['close'] < previous['close']:
                bounce_percent = (previous['close'] - current['close']) / previous['close']
                if bounce_percent > self.config['price_bounce']:
                    signal = -1
                    signal_strength += 0.5
                    signal_reasons.append(f"BB上限からの反発({bounce_percent*100:.2f}%)")
        
        # 3. 価格-MA乖離に基づくシグナル
        if 'SMA_short' in current and signal == 0:
            # 短期移動平均からの大きな乖離を検出
            deviation = (current['close'] - current['SMA_short']) / current['SMA_short']
            
            if deviation < -self.config['ma_deviation']:  # 短期MAから下方乖離
                signal = 1
                signal_strength += abs(deviation) * 30
                signal_reasons.append(f"MA乖離({deviation*100:.2f}%)")
            elif deviation > self.config['ma_deviation']:  # 短期MAから上方乖離
                signal = -1
                signal_strength += abs(deviation) * 30
                signal_reasons.append(f"MA乖離({deviation*100:.2f}%)")
        
        # 4. RSIの過度な値からの反転を検出
        if 'RSI' in current and 'RSI' in previous and signal == 0:
            # RSIの反転を検出
            if previous['RSI'] < self.config['rsi_extreme_lower'] and current['RSI'] > previous['RSI']:
                # オーバーソールドからの反転
                signal = 1
                signal_strength += 0.3
                signal_reasons.append(f"RSI反転(現在:{current['RSI']:.1f}, 前回:{previous['RSI']:.1f})")
            elif previous['RSI'] > self.config['rsi_extreme_upper'] and current['RSI'] < previous['RSI']:
                # オーバーボウトからの反転
                signal = -1
                signal_strength += 0.3
                signal_reasons.append(f"RSI反転(現在:{current['RSI']:.1f}, 前回:{previous['RSI']:.1f})")
        
        # 5. MACD ヒストグラムの反転を確認
        if 'MACD_hist' in current and 'MACD_hist_prev' in current and signal != 0:
            # ヒストグラムが0を上に超えた（買い）
            if current['MACD_hist'] > 0 and current['MACD_hist_prev'] <= 0 and signal > 0:
                signal_strength += 0.5
                signal_reasons.append("MACDヒストグラム反転(上向き)")
            # ヒストグラムが0を下に超えた（売り）
            elif current['MACD_hist'] < 0 and current['MACD_hist_prev'] >= 0 and signal < 0:
                signal_strength += 0.5
                signal_reasons.append("MACDヒストグラム反転(下向き)")
        
        # Z-スコア計算（標準偏差を使った偏差の測定）
        price_series = data['close'].tail(self.config['lookback_period'])
        mean_price = price_series.mean()
        std_price = price_series.std()
        z_score = (current['close'] - mean_price) / std_price if std_price > 0 else 0
        
        # Z-スコアによるフィルタリング
        if abs(z_score) < self.config['z_score_threshold']:
            # z_score が小さすぎる場合はシグナルをキャンセル
            # ただし、すでに記録されたシグナルの理由は保持
            tmp_signal = signal
            signal = 0
        else:
            # z_score の大きさに基づいてシグナル強度を加算
            signal_strength += min(abs(z_score) * 0.2, 0.6)
            signal_reasons.append(f"Z-score:{z_score:.2f}")
        
        # トレンドとの一致確認
        if 'ma_signal' in current:
            # 買いシグナルなのに下降トレンドの場合
            if signal > 0 and current['ma_signal'] < 0:
                # より強い平均回帰シグナルの場合、トレンドを無視できる
                if signal_strength < 0.7:  # 強度が不十分な場合
                    signal = 0
                else:
                    signal_reasons.append("トレンド逆行(強シグナル)")
            
            # 売りシグナルなのに上昇トレンドの場合
            if signal < 0 and current['ma_signal'] > 0:
                # より強い平均回帰シグナルの場合、トレンドを無視できる
                if signal_strength < 0.7:  # 強度が不十分な場合
                    signal = 0
                else:
                    signal_reasons.append("トレンド逆行(強シグナル)")
        
        # ボラティリティフィルター
        atr_ratio = current.get('ATR', current['close'] * 0.01) / current['close']
        
        # 閾値に基づくフィルタリング
        if atr_ratio > self.config['vol_high_threshold']:
            # 高ボラティリティ環境では強力なシグナルのみ許可
            if signal_strength < 0.8:
                signal = 0
            else:
                signal_reasons.append(f"高ボラティリティ({atr_ratio*100:.2f}%)")
        
        # 反対に、ボラティリティが極端に低い場合もシグナルを弱める
        if atr_ratio < self.config['vol_low_threshold']:
            # 極端に低いボラティリティでは強いシグナルが必要
            if signal_strength < 0.6:
                signal = 0
            else:
                signal_reasons.append(f"低ボラティリティ({atr_ratio*100:.2f}%)")
        
        # シグナル情報をまとめる
        signal_info = {
            'timestamp': current['timestamp'],
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'signal': signal,
            'mean_price': mean_price,
            'std_price': std_price,
            'z_score': z_score,
            'atr_ratio': atr_ratio,
            'signal_strength': signal_strength,
            'signal_reasons': signal_reasons,
            'mean_reversion_detected': True if signal != 0 else False
        }
        
        # 既存の指標情報も追加
        for key in ['RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'ma_signal']:
            if key in current:
                signal_info[key] = current[key]
        
        return signal_info
    
    def calculate_risk_reward(self, signal_info, base_sl_percent=1.5, base_tp_percent=5.0):
        """
        Mean Reversion戦略に特化したリスク/リワード設定を計算
        
        Parameters:
        -----------
        signal_info : dict
            シグナル情報
        base_sl_percent : float
            基本のストップロス割合（％）
        base_tp_percent : float
            基本のテイクプロフィット割合（％）
            
        Returns:
        --------
        tuple
            (stop_loss_percent, take_profit_percent)
        """
        # 基本設定
        sl_percent = base_sl_percent
        tp_percent = base_tp_percent
        
        # シグナル強度による調整
        signal_strength = signal_info.get('signal_strength', 0.5)
        
        if signal_strength > 0.7:  # 強いシグナル
            # 強いシグナルではより狭いTP（早めの利確）と狭いSL（小さなリスク）
            tp_percent = tp_percent * 0.7
            sl_percent = sl_percent * 0.7
        else:
            # 通常のシグナル
            tp_percent = tp_percent * 0.8
            sl_percent = sl_percent * 0.8
        
        # ボラティリティに基づく調整
        atr_ratio = signal_info.get('atr_ratio', 0.01)
        
        if atr_ratio < 0.005:  # 非常に低いボラティリティ
            # 極端に狭いSL/TP
            sl_percent *= 0.7
            tp_percent *= 0.7
        elif atr_ratio < 0.01:  # 低〜中ボラティリティ
            # 最適環境 - 標準設定
            pass
        else:  # 高ボラティリティ
            # より広いSL（誤発動防止）、狭いTP（素早く利益確定）
            sl_percent *= 1.2
            tp_percent *= 0.9
        
        # Z-スコアによる調整
        z_score = abs(signal_info.get('z_score', 0))
        if z_score > 2.5:  # 極端な乖離
            # より狭いTP（早期に利益確定）
            tp_percent *= 0.8
        
        # 最終値を妥当な範囲に収める
        sl_percent = max(0.8, min(sl_percent, 2.5))
        tp_percent = max(1.5, min(tp_percent, 8.0))
        
        # 最低リスク/リワード比の調整（平均回帰では低めに設定）
        min_risk_reward = 1.5  # 通常の2.0より低い
        
        # 最低リスク/リワード比の保証
        if tp_percent / sl_percent < min_risk_reward:
            tp_percent = sl_percent * min_risk_reward
        
        return sl_percent, tp_percent