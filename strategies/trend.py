# -*- coding: utf-8 -*-
"""
最適化されたトレンドフォロー戦略モジュール

処理速度を大幅に改善した実装です。
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
            'adx_strong_threshold': 40, # ADX非常に強いトレンド閾値
            'ma_trend_length': 50,     # 長期トレンド確認用MA期間
            'min_signal_strength': 0.4, # 最小シグナル強度 (0.5から0.4に緩和)
            'vol_filter': True,        # ボラティリティフィルター
            'high_vol_threshold': 0.02, # 高ボラティリティ閾値
            'confirmation_needed': True, # 複数確認要素の必要性
            'use_caching': True,        # キャッシングを使用（新規追加）
            'cache_update_interval': 20  # キャッシュ更新間隔（新規追加）
        }
        
        # 設定の上書き
        if config:
            self.config.update(config)
            
        # キャッシュ初期化
        self._adx_cache = {}  # ADXキャッシュ
    
    def generate_signals(self, data):
        """
        トレンドフォロー戦略に基づくトレーディングシグナルを生成（最適化版）
        
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
        confirmation_count = 0  # 確認要素のカウント
        
        # 1. 移動平均クロスオーバー検出 - 実装はそのまま
        if 'EMA_short' in current and 'EMA_long' in current:
            # 短期MAが長期MAを上抜け（買い）
            if (previous['EMA_short'] <= previous['EMA_long'] and 
                current['EMA_short'] > current['EMA_long']):
                signal = 1
                # クロスオーバーの強さを計算（乖離率）
                crossover_strength = (current['EMA_short'] / current['EMA_long'] - 1) * 100
                signal_strength += min(crossover_strength * 5, 0.5)  # 最大0.5
                signal_reasons.append(f"EMAクロス(上抜け:{crossover_strength:.2f}%)")
                confirmation_count += 1
            
            # 短期MAが長期MAを下抜け（売り）
            elif (previous['EMA_short'] >= previous['EMA_long'] and 
                  current['EMA_short'] < current['EMA_long']):
                signal = -1
                # クロスオーバーの強さを計算
                crossover_strength = (1 - current['EMA_short'] / current['EMA_long']) * 100
                signal_strength += min(crossover_strength * 5, 0.5)  # 最大0.5
                signal_reasons.append(f"EMAクロス(下抜け:{crossover_strength:.2f}%)")
                confirmation_count += 1
        
        # 2. MACD シグナルクロス - 実装はそのまま
        if 'MACD' in current and 'MACD_signal' in current and signal != 0:
            # MACDがシグナルラインを上抜け（買い）
            if (previous['MACD'] <= previous['MACD_signal'] and 
                current['MACD'] > current['MACD_signal']):
                # 買いシグナル強化
                if signal > 0:
                    # クロスの強さを計算
                    macd_strength = abs(current['MACD'] - current['MACD_signal']) / abs(current['MACD_signal']) if current['MACD_signal'] != 0 else 0
                    signal_strength += min(macd_strength * 10, 0.6)  # 最大0.6
                    signal_reasons.append(f"MACDクロス(上向き)")
                    confirmation_count += 1
                    
                    # ヒストグラムが加速しているかチェック
                    if 'MACD_hist' in current and 'MACD_hist_prev' in current:
                        if current['MACD_hist'] > current['MACD_hist_prev']:
                            signal_strength += 0.1
                            signal_reasons[-1] += "(加速)"
                            confirmation_count += 0.5
                else:
                    # 買いMACDだが売りシグナルの場合はシグナルを弱める
                    signal_strength *= 0.7
                    signal_reasons.append("MACDと方向不一致")
            
            # MACDがシグナルラインを下抜け（売り）
            elif (previous['MACD'] >= previous['MACD_signal'] and 
                  current['MACD'] < current['MACD_signal']):
                # 売りシグナル強化
                if signal < 0:
                    # クロスの強さを計算
                    macd_strength = abs(current['MACD'] - current['MACD_signal']) / abs(current['MACD_signal']) if current['MACD_signal'] != 0 else 0
                    signal_strength += min(macd_strength * 10, 0.6)  # 最大0.6
                    signal_reasons.append(f"MACDクロス(下向き)")
                    confirmation_count += 1
                    
                    # ヒストグラムが加速しているかチェック
                    if 'MACD_hist' in current and 'MACD_hist_prev' in current:
                        if current['MACD_hist'] < current['MACD_hist_prev']:
                            signal_strength += 0.1
                            signal_reasons[-1] += "(加速)"
                            confirmation_count += 0.5
                else:
                    # 売りMACDだが買いシグナルの場合はシグナルを弱める
                    signal_strength *= 0.7
                    signal_reasons.append("MACDと方向不一致")
        
        # 3. RSIによる過剰売買の検出と修正
        if 'RSI' in current and signal != 0:
            # 買いシグナルだがRSIが過買い状態
            if signal > 0 and current['RSI'] > self.config['rsi_overbought']:
                # 過買い状態なのでシグナルを弱める
                signal_strength *= 0.6  # 0.5に変更
                signal_reasons.append(f"RSI過買い警告({current['RSI']:.1f})")
            
            # 売りシグナルだがRSIが過売り状態
            elif signal < 0 and current['RSI'] < self.config['rsi_oversold']:
                # 過売り状態なのでシグナルを弱める
                signal_strength *= 0.6  # 0.5に変更
                signal_reasons.append(f"RSI過売り警告({current['RSI']:.1f})")
            
            # トレンド方向と一致するRSI
            elif (signal > 0 and current['RSI'] > 50 and current['RSI'] < 70) or (signal < 0 and current['RSI'] < 50 and current['RSI'] > 30):
                # トレンドとRSIが一致し、極端な値ではない場合はシグナルを強化
                signal_strength += 0.15  # 0.1から0.15へ
                signal_reasons.append(f"RSIトレンド一致({current['RSI']:.1f})")
                confirmation_count += 1
        
        # 4. ADXによるトレンド強度の確認 - キャッシュ機能を追加
        adx_value = self._calculate_adx(data)
        
        # ADXが強いトレンドを示す場合、シグナル強度を加算
        if adx_value > self.config['adx_strong_threshold']:  # 非常に強いトレンド
            signal_strength += 0.2
            signal_reasons.append(f"ADX強トレンド({adx_value:.1f})")
            confirmation_count += 1
        elif adx_value > self.config['adx_threshold']:  # 強いトレンド
            signal_strength += 0.1
            signal_reasons.append(f"ADXトレンド({adx_value:.1f})")
            confirmation_count += 0.5
        elif adx_value < 20 and signal != 0:  # 弱いトレンド - シグナルをキャンセル
            old_signal = signal
            signal = 0
            signal_reasons.append(f"ADX弱すぎ({adx_value:.1f})")
        
        # 5. 価格とMAの位置関係による確認
        if 'SMA_short' in current and 'SMA_long' in current and signal != 0:
            price_vs_ma = 0
            
            # 買いシグナルの場合、価格が両方のMAより上にあればトレンド強化
            if signal > 0:
                if current['close'] > max(current['SMA_short'], current['SMA_long']):
                    price_vs_ma = 1
                    signal_reasons.append("価格>両MA")
                    confirmation_count += 1
                elif current['close'] < min(current['SMA_short'], current['SMA_long']):
                    price_vs_ma = -1
                    signal_reasons.append("価格<両MA")
            
            # 売りシグナルの場合、価格が両方のMAより下にあればトレンド強化
            elif signal < 0:
                if current['close'] < min(current['SMA_short'], current['SMA_long']):
                    price_vs_ma = 1
                    signal_reasons.append("価格<両MA")
                    confirmation_count += 1
                elif current['close'] > max(current['SMA_short'], current['SMA_long']):
                    price_vs_ma = -1
                    signal_reasons.append("価格>両MA")
            
            # シグナル強度を調整
            if price_vs_ma == 1:  # トレンドを強化
                signal_strength += 0.15  # 0.1から0.15へ
            elif price_vs_ma == -1:  # トレンドに反するので強度を下げる
                signal_strength *= 0.7
        
        # 6. ボラティリティフィルター
        if 'ATR' in current:
            atr_ratio = current['ATR'] / current['close']
            
            # 極端に低いボラティリティはトレンドの信頼性を下げる
            if atr_ratio < 0.005 and signal != 0:
                signal_strength *= 0.6  # 0.7から0.6へさらに弱める
                signal_reasons.append(f"低ボラティリティ({atr_ratio*100:.2f}%)")
            
            # 高ボラティリティは良いが、過度な場合は注意
            elif atr_ratio > self.config['high_vol_threshold'] and signal != 0:
                signal_strength *= 0.6  # 0.8から0.6へさらに弱める
                signal_reasons.append(f"過度なボラティリティ({atr_ratio*100:.2f}%)")
            elif atr_ratio > 0.01 and atr_ratio < self.config['high_vol_threshold'] and signal != 0:
                signal_strength += 0.15  # 0.1から0.15へ
                signal_reasons.append(f"適正ボラティリティ({atr_ratio*100:.2f}%)")
                confirmation_count += 0.5
        
        # 7. 長期トレンド確認 - 閾値を緩和
        if len(data) > self.config['ma_trend_length'] and signal != 0:
            # 長期MAの計算
            long_term_ma = data['close'].rolling(window=self.config['ma_trend_length']).mean().iloc[-1]
            
            # 価格と長期MAの位置関係を確認
            price_ma_diff = (current['close'] - long_term_ma) / long_term_ma
            
            # 閾値を0.5%から0.2%に緩和
            if signal > 0 and price_ma_diff > 0.002:  # 0.2%以上上にある
                signal_strength += 0.2
                signal_reasons.append(f"長期トレンド一致(上昇)")
                confirmation_count += 1
            elif signal < 0 and price_ma_diff < -0.002:  # 0.2%以上下にある
                signal_strength += 0.2
                signal_reasons.append(f"長期トレンド一致(下降)")
                confirmation_count += 1
            elif (signal > 0 and price_ma_diff < -0.005) or (signal < 0 and price_ma_diff > 0.005):
                # 明確に逆方向の場合のみ無効化
                old_signal = signal
                signal = 0
                signal_reasons.append(f"長期トレンド不一致(差:{price_ma_diff*100:.2f}%)")
        
        # 8. 複数確認要素の確認
        if self.config['confirmation_needed'] and signal != 0:
            # 確認要素の閾値を緩和（2から1.5に）
            if confirmation_count < 1.5:  # 複数の確認要素が必要
                old_signal = signal
                signal = 0
                signal_reasons.append(f"確認不足(確認数:{confirmation_count})")
        
        # 最終的なシグナル強度が閾値を超えない場合、シグナルを無効化
        if signal_strength < self.config['min_signal_strength'] and signal != 0:
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
            'confirmation_count': confirmation_count,
            'signal_reasons': signal_reasons
        }
        
        # 既存の指標情報も追加
        for key in ['RSI', 'MACD', 'SMA_short', 'SMA_long', 'ATR']:
            if key in current:
                signal_info[key] = current[key]
        
        return signal_info
    
    def _calculate_adx(self, data, period=14):
        """
        最適化されたADX（平均方向性指数）の計算
        
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
        # データフレームに既にADXが計算されていれば再利用
        if 'ADX' in data.columns:
            return data['ADX'].iloc[-1]
        
        # キャッシュを使用する場合
        if self.config.get('use_caching', True):
            cache_key = len(data)
            if cache_key in self._adx_cache:
                return self._adx_cache[cache_key]
        
        # 最適化: numpy配列を使用
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # 計算効率化: numpy配列で計算
        tr = np.zeros(len(high))
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        
        # First Value
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], 
                        abs(high[i] - close[i-1]), 
                        abs(low[i] - close[i-1]))
            
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
                
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # 移動平均計算
        tr_ma = np.zeros(len(tr))
        plus_dm_ma = np.zeros(len(plus_dm))
        minus_dm_ma = np.zeros(len(minus_dm))
        
        # 最初のピリオドの平均
        tr_ma[period] = tr[1:period+1].mean()
        plus_dm_ma[period] = plus_dm[1:period+1].mean()
        minus_dm_ma[period] = minus_dm[1:period+1].mean()
        
        # 残りのピリオドの計算を最適化
        for i in range(period+1, len(tr)):
            tr_ma[i] = tr_ma[i-1] - (tr_ma[i-1] / period) + tr[i]
            plus_dm_ma[i] = plus_dm_ma[i-1] - (plus_dm_ma[i-1] / period) + plus_dm[i]
            minus_dm_ma[i] = minus_dm_ma[i-1] - (minus_dm_ma[i-1] / period) + minus_dm[i]
        
        # +DI, -DI
        plus_di = np.zeros(len(tr_ma))
        minus_di = np.zeros(len(tr_ma))
        
        for i in range(period, len(tr_ma)):
            if tr_ma[i] > 0:
                plus_di[i] = 100 * plus_dm_ma[i] / tr_ma[i]
                minus_di[i] = 100 * minus_dm_ma[i] / tr_ma[i]
        
        # DX
        dx = np.zeros(len(plus_di))
        
        for i in range(period, len(plus_di)):
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX
        adx = np.zeros(len(dx))
        
        # 最初のADX値
        adx[2*period-1] = dx[period:2*period].mean()
        
        # 残りのADX値
        for i in range(2*period, len(adx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        # 最終値
        adx_value = adx[-1] if adx[-1] > 0 else 0
        
        # キャッシュに保存
        if self.config.get('use_caching', True):
            self._adx_cache[cache_key] = adx_value
        
        return adx_value