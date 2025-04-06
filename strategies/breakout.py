# -*- coding: utf-8 -*-
"""
最適化されたブレイクアウト戦略モジュール

処理時間の大幅な改善を目指しています。
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
            'lookback_period': 20,         # サポート/レジスタンス検出期間
            'breakout_threshold': 0.005,   # ブレイクアウト確認閾値 (0.2%から0.5%に引き上げ)
            'volume_factor': 2.0,          # 出来高の通常比 (1.5から2.0に引き上げ)
            'min_range_percent': 0.02,     # 最小レンジ幅（％）
            'vol_filter_threshold': 0.005, # 極端な低ボラティリティ閾値
            'adx_threshold': 25,           # トレンド確認のADX閾値（新規追加）
            'confirmation_candles': 1,      # 確認用キャンドル数（新規追加）
            'false_breakout_filter': True,  # 偽ブレイクアウトフィルター（新規追加）
            'min_signal_strength': 0.6,     # 最小シグナル強度（新規追加）
            'important_level_only': True,   # 重要なレベルのみ利用（新規追加）
            'use_caching': True,            # キャッシングを使用（新規追加）
            'cache_update_interval': 20     # キャッシュ更新間隔（新規追加）
        }
        
        # 設定の上書き
        if config:
            self.config.update(config)
            
        # キャッシュ初期化
        self._sr_cache = {}  # サポート/レジスタンスキャッシュ
        self._adx_cache = {}  # ADXキャッシュ
        self._atr_cache = {}  # ATRキャッシュ
        self._level_importance_cache = {}  # レベル重要度キャッシュ
    
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
        
        # 確認用キャンドル数を考慮（遅延シグナル）
        confirm_idx = -1 - self.config['confirmation_candles']
        if abs(confirm_idx) > len(data):
            return {}
            
        previous = data.iloc[confirm_idx - 1]
        confirmed = data.iloc[confirm_idx]
        
        # サポート・レジスタンスを計算
        support, resistance = self._calculate_support_resistance(data.iloc[:confirm_idx])
        
        # 基本シグナル
        signal = 0
        signal_strength = 0
        signal_reasons = []
        
        # ATRの計算（ボラティリティ測定用）- キャッシュを使用
        atr = self._calculate_atr(data)
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
        
        # ADXの計算（トレンド強度測定用）- キャッシュを使用
        adx_value = self._calculate_adx(data)
        is_trending = adx_value > self.config['adx_threshold']
        
        # 重要なレベルかどうかの判定 - キャッシュを使用
        is_important_level = True
        if self.config['important_level_only']:
            # キャッシュキー
            cache_key = (len(data), round(support, 2) if support else 0, round(resistance, 2) if resistance else 0)
            
            # キャッシュにあれば使用
            if cache_key in self._level_importance_cache:
                is_important_level = self._level_importance_cache[cache_key]
            else:
                # 重要なレベルかどうかを判定
                is_important_level = self._is_important_level(data, support, resistance)
                # キャッシュに保存
                self._level_importance_cache[cache_key] = is_important_level
            
            # 重要なレベルでない場合はシグナルを生成しない
            if not is_important_level:
                return {
                    'timestamp': current['timestamp'],
                    'signal': 0,
                    'signal_strength': 0,
                    'signal_reasons': ["重要度の低いレベル"]
                }
        
        # レジスタンスブレイクアウト（買い）
        if resistance is not None:
            # 前回の高値がレジスタンスを下回り、確認用キャンドルの終値がレジスタンスを上回っている場合
            if previous['high'] <= resistance and confirmed['close'] > resistance:
                # 確認：十分な上昇勢い（より確実なブレイクアウト）
                breakout_percent = (confirmed['close'] / resistance - 1)
                if breakout_percent >= self.config['breakout_threshold']:
                    # トレンド環境のみブレイクアウト有効（トレンド方向と一致）
                    if not is_trending or (is_trending and self._get_trend_direction(data) > 0):
                        signal = 1
                        signal_strength = min(breakout_percent * 100, 0.7)  # 最大0.7
                        signal_reasons.append(f"レジスタンスブレイク({breakout_percent*100:.2f}%上抜け)")
                        
                        # 出来高増加で確認
                        if 'volume' in confirmed and 'volume_ma' in confirmed:
                            if confirmed['volume'] > confirmed['volume_ma'] * self.config['volume_factor']:
                                signal_strength += 0.3
                                signal_reasons.append(f"高出来高確認({confirmed['volume']/confirmed['volume_ma']:.1f}倍)")
                            else:
                                signal_strength *= 0.8
                                signal_reasons.append("出来高不足")
                        
                        # 偽ブレイクアウトフィルター
                        if self.config['false_breakout_filter']:
                            if self._is_false_breakout(data, confirmed, resistance, is_support=False):
                                signal = 0
                                signal_reasons.append("偽ブレイクアウト検出")
        
        # サポートブレイクダウン（売り）
        if support is not None and signal == 0:
            # 前回の安値がサポートを上回り、確認用キャンドルの終値がサポートを下回っている場合
            if previous['low'] >= support and confirmed['close'] < support:
                # 確認：十分な下落勢い
                breakdown_percent = (1 - confirmed['close'] / support)
                if breakdown_percent >= self.config['breakout_threshold']:
                    # トレンド環境のみブレイクアウト有効（トレンド方向と一致）
                    if not is_trending or (is_trending and self._get_trend_direction(data) < 0):
                        signal = -1
                        signal_strength = min(breakdown_percent * 100, 0.7)  # 最大0.7
                        signal_reasons.append(f"サポートブレイク({breakdown_percent*100:.2f}%下抜け)")
                        
                        # 出来高増加で確認
                        if 'volume' in confirmed and 'volume_ma' in confirmed:
                            if confirmed['volume'] > confirmed['volume_ma'] * self.config['volume_factor']:
                                signal_strength += 0.3
                                signal_reasons.append(f"高出来高確認({confirmed['volume']/confirmed['volume_ma']:.1f}倍)")
                            else:
                                signal_strength *= 0.8
                                signal_reasons.append("出来高不足")
                        
                        # 偽ブレイクアウトフィルター
                        if self.config['false_breakout_filter']:
                            if self._is_false_breakout(data, confirmed, support, is_support=True):
                                signal = 0
                                signal_reasons.append("偽ブレイクアウト検出")
        
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
                signal_strength *= 0.7
                signal_reasons.append("トレンド逆行")
        
        # シグナル強度が最小閾値を下回る場合はシグナルを無効化
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
            'atr_ratio': atr_ratio,
            'adx': adx_value,
            'signal_strength': signal_strength,
            'resistance': resistance,
            'support': support,
            'breakout_detected': True if signal != 0 else False,
            'is_important_level': is_important_level,
            'signal_reasons': signal_reasons
        }
        
        # 既存の指標情報も追加
        for key in ['RSI', 'SMA_short', 'SMA_long']:
            if key in current:
                signal_info[key] = current[key]
        
        return signal_info
    
    def _calculate_support_resistance(self, data):
        """
        サポート・レジスタンスレベルの計算（キャッシュ機能付き）
        
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
        
        # キャッシュを使用する場合
        if self.config.get('use_caching', True):
            cache_key = len(data)
            
            # キャッシュに値があり、更新間隔内なら再利用
            if cache_key in self._sr_cache:
                last_update, support, resistance = self._sr_cache[cache_key]
                if cache_key - last_update < self.config.get('cache_update_interval', 20):
                    return support, resistance
        
        # キャッシュになければ計算
        lookback = self.config['lookback_period']
        recent_data = data.iloc[-lookback:]
        
        # スイングハイ・ローの検出
        swing_highs = []
        swing_lows = []
        
        # 最適化: 一度に配列を走査
        for i in range(2, len(recent_data) - 2):
            current_high = recent_data['high'].iloc[i]
            current_low = recent_data['low'].iloc[i]
            
            # 前後のデータをローカル変数に保存（高速化）
            prev1_high = recent_data['high'].iloc[i-1]
            prev2_high = recent_data['high'].iloc[i-2]
            next1_high = recent_data['high'].iloc[i+1]
            next2_high = recent_data['high'].iloc[i+2]
            
            prev1_low = recent_data['low'].iloc[i-1]
            prev2_low = recent_data['low'].iloc[i-2]
            next1_low = recent_data['low'].iloc[i+1]
            next2_low = recent_data['low'].iloc[i+2]
            
            # ローカルトップ（スイングハイ）の検出
            if (current_high > prev1_high and current_high > prev2_high and
                current_high > next1_high and current_high > next2_high):
                swing_highs.append(current_high)
            
            # ローカルボトム（スイングロー）の検出
            if (current_low < prev1_low and current_low < prev2_low and
                current_low < next1_low and current_low < next2_low):
                swing_lows.append(current_low)
        
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
        
        # キャッシュに結果を保存
        if self.config.get('use_caching', True):
            self._sr_cache[cache_key] = (cache_key, support, resistance)
        
        return support, resistance
    
    def _calculate_atr(self, data, period=14):
        """
        ATR（Average True Range）の計算（キャッシュ機能付き）
        
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
        # データフレームに既に計算済みならそれを使用
        if 'ATR' in data.columns:
            return data['ATR'].iloc[-1]
        
        # キャッシュを使用する場合
        if self.config.get('use_caching', True):
            cache_key = len(data)
            if cache_key in self._atr_cache:
                return self._atr_cache[cache_key]
        
        # 効率化: 必要な列のみを抽出
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        
        # 最大値を取得 (効率化: pandas concat は重いのでnumpy max を使用)
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(tr).rolling(period).mean().iloc[-1]
        
        # キャッシュに結果を保存
        if self.config.get('use_caching', True):
            self._atr_cache[cache_key] = atr if not pd.isna(atr) else data['close'].iloc[-1] * 0.01
        
        return atr if not pd.isna(atr) else data['close'].iloc[-1] * 0.01  # デフォルト：価格の1%
    
    def _calculate_adx(self, data, period=14):
        """
        ADX（平均方向性指数）の計算（キャッシュ機能付き）
        
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
        # データフレームに既に計算済みなら使用
        if 'ADX' in data.columns and not pd.isna(data['ADX'].iloc[-1]):
            return data['ADX'].iloc[-1]
        
        # キャッシュを使用する場合
        if self.config.get('use_caching', True):
            cache_key = len(data)
            if cache_key in self._adx_cache:
                return self._adx_cache[cache_key]
            
        # 効率化: 必要な列のみを抽出して計算
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # 計算効率化: numpy配列で計算
        tr = np.zeros(len(high))
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        
        for i in range(1, len(high)):
            # TR
            tr[i] = max(high[i] - low[i], 
                        abs(high[i] - close[i-1]), 
                        abs(low[i] - close[i-1]))
            
            # +DM, -DM
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
        
        # 残りのピリオド
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
        
        # キャッシュに結果を保存
        if self.config.get('use_caching', True):
            self._adx_cache[cache_key] = adx_value
        
        return adx_value
        
    def _get_trend_direction(self, data, period=14):
        """
        トレンド方向の判定（最適化版）
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データ
        period : int
            判定期間
            
        Returns:
        --------
        int
            トレンド方向 (1: 上昇, -1: 下降, 0: レンジ)
        """
        if len(data) < period:
            return 0
            
        # シンプルなトレンド判定 - 移動平均の傾き
        if 'SMA_short' in data.columns:
            ma_start = data['SMA_short'].iloc[-period]
            ma_end = data['SMA_short'].iloc[-1]
            
            # 傾きの判定
            ma_slope = (ma_end - ma_start) / ma_start
            
            if ma_slope > 0.01:  # 1%以上の上昇
                return 1
            elif ma_slope < -0.01:  # 1%以上の下降
                return -1
            else:
                return 0
        
        # 移動平均がない場合は終値を使用
        else:
            close_start = data['close'].iloc[-period]
            close_end = data['close'].iloc[-1]
            
            close_slope = (close_end - close_start) / close_start
            
            if close_slope > 0.01:
                return 1
            elif close_slope < -0.01:
                return -1
            else:
                return 0
    
    def _is_important_level(self, data, support, resistance):
        """
        サポート/レジスタンスが重要なレベルかどうかを判定（最適化版）
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データ
        support : float
            サポートレベル
        resistance : float
            レジスタンスレベル
            
        Returns:
        --------
        bool
            重要なレベルかどうか
        """
        if len(data) < 30:
            return True
            
        # 最適化: 限られたデータポイントのみ分析
        max_lookback = min(100, len(data))  # 最大100キャンドルに制限
        recent_data = data.iloc[-max_lookback:]
        
        # 簡易版: サポート/レジスタンスレベルの周辺（±0.5%）でのタッチ回数をカウント
        touch_threshold = 0.005  # レベルの±0.5%以内
        min_touches = 2  # 最低2回のタッチが必要
        
        support_touches = 0
        resistance_touches = 0
        
        # 効率化: 必要な計算のみ行う
        if support is not None:
            support_min = support * (1 - touch_threshold)
            support_max = support * (1 + touch_threshold)
            
            # ベクトル化演算
            support_touched = (recent_data['low'] >= support_min) & (recent_data['low'] <= support_max)
            support_touches = support_touched.sum()
            
            # 早期リターン
            if support_touches >= min_touches:
                return True
            
        if resistance is not None:
            resistance_min = resistance * (1 - touch_threshold)
            resistance_max = resistance * (1 + touch_threshold)
            
            # ベクトル化演算
            resistance_touched = (recent_data['high'] >= resistance_min) & (recent_data['high'] <= resistance_max)
            resistance_touches = resistance_touched.sum()
            
            # 早期リターン
            if resistance_touches >= min_touches:
                return True
        
        # いずれかのレベルが最低タッチ回数を満たしていれば重要
        return (support_touches >= min_touches) or (resistance_touches >= min_touches)
    
    def _is_false_breakout(self, data, confirmed_candle, level, is_support=True):
        """
        偽ブレイクアウトを検出（最適化版）
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データ
        confirmed_candle : pandas.Series
            確認用キャンドル
        level : float
            ブレイクアウトレベル
        is_support : bool
            サポートレベルかどうか
            
        Returns:
        --------
        bool
            偽ブレイクアウトかどうか
        """
        # 効率化: 早期判定のために重要度チェックを追加
        # 1. ブレイクアウトの大きさをチェック - より大きなブレイクアウトを認める
        if is_support:
            # サポートブレイク (下方向)
            breakout_size = (level - confirmed_candle['close']) / level
            # 大きなブレイクアウトはより信頼性が高い
            if breakout_size > self.config['breakout_threshold'] * 2.5:
                return False  # 偽ブレイクアウトではない
            # 小さなブレイクアウト
            if breakout_size < self.config['breakout_threshold'] * 1.2:
                return True  # 偽ブレイクアウトの可能性が高い
        else:
            # レジスタンスブレイク (上方向)
            breakout_size = (confirmed_candle['close'] - level) / level
            # 大きなブレイクアウトはより信頼性が高い 
            if breakout_size > self.config['breakout_threshold'] * 2.5:
                return False  # 偽ブレイクアウトではない
            # 小さなブレイクアウト
            if breakout_size < self.config['breakout_threshold'] * 1.2:
                return True  # 偽ブレイクアウトの可能性が高い
        
        # 2. 出来高チェック - 効率化のため条件チェックを簡素化
        if 'volume' in confirmed_candle and 'volume_ma' in confirmed_candle:
            if confirmed_candle['volume'] < confirmed_candle['volume_ma'] * 1.2:
                return True
        
        # 3. 後続キャンドルでの価格動向チェック
        # (本来は数キャンドル後まで見るべきだが、現在のデータ構造では直後のキャンドルのみ確認)
        
        # デフォルトは偽ブレイクアウトでない
        return False