# -*- coding: utf-8 -*-
"""
最適化された戦略統合モジュール

処理時間を大幅に短縮し、Mean Reversion戦略の性能を保護しています。
"""
import numpy as np
import pandas as pd

class StrategyIntegrator:
    def __init__(self, config=None):
        """
        戦略統合クラスの初期化
        
        Parameters:
        -----------
        config : dict, optional
            設定パラメータ
        """
        # デフォルト設定
        self.config = {
            'buy_threshold': 0.25,     # 買いシグナル閾値
            'sell_threshold': -0.25,   # 売りシグナル閾値
            'adx_threshold': 25,       # トレンド判定のADX閾値
            'high_vol_threshold': 0.012, # 高ボラティリティ閾値
            'low_vol_threshold': 0.006,  # 低ボラティリティ閾値
            'strong_signal_threshold': 0.7, # 強いシグナル閾値
            'very_strong_signal_threshold': 0.8, # 非常に強いシグナル閾値
            'preserve_mean_reversion': True,  # Mean Reversion戦略の重み保持
            'trend_quality_filter': True,     # トレンド品質フィルター
            'breakout_quality_filter': True,  # ブレイクアウト品質フィルター
            'use_caching': True,              # キャッシング機能を使用（新規追加）
            'mr_max_hold_hours': 12,          # Mean Reversion最大保有時間（新規追加）
            'mr_tp_factor': 0.65              # Mean Reversionの早期利益確定係数（新規追加）
        }
        
        # 設定の上書き
        if config:
            self.config.update(config)
            
        # キャッシュ機構の初期化
        self._adx_cache = {}
        self._weight_scenarios_cache = {}
    
    def integrate_strategies(self, trend_signal, breakout_signal, mean_reversion_signal, data):
        """
        複数の取引戦略からシグナルを統合（最適化版）
        
        Parameters:
        -----------
        trend_signal : dict
            トレンドフォロー戦略からのシグナル情報
        breakout_signal : dict
            ブレイクアウト戦略からのシグナル情報
        mean_reversion_signal : dict
            平均回帰戦略からのシグナル情報
        data : pandas.DataFrame
            最新の価格データと指標
            
        Returns:
        --------
        dict
            統合されたシグナル情報
        """
        if data.empty:
            return {}
        
        # 最新のデータポイント
        current = data.iloc[-1]
        
        # シグナルと信頼度スコア
        signals = {
            'trend': trend_signal.get('signal', 0),
            'breakout': breakout_signal.get('signal', 0),
            'mean_reversion': mean_reversion_signal.get('signal', 0)
        }
        
        # シグナル強度を考慮
        signal_strengths = {
            'trend': trend_signal.get('signal_strength', 0.5),
            'breakout': breakout_signal.get('signal_strength', 0.5),
            'mean_reversion': mean_reversion_signal.get('signal_strength', 0.5)
        }
        
        # 市場環境の分析（トレンド/レンジ判定） - キャッシュ使用
        adx_value = self._calculate_adx(data)
        is_trending = adx_value > self.config['adx_threshold']
        
        # ボラティリティ評価
        atr = current.get('ATR', current['close'] * 0.01)
        atr_ratio = atr / current['close']
        is_high_volatility = atr_ratio > self.config['high_vol_threshold']
        is_low_volatility = atr_ratio < self.config['low_vol_threshold']
        
        # 最適化: MR戦略が単独の場合、優先処理
        if self.config['preserve_mean_reversion']:
            # Mean Reversionのシグナルが単独で発生した場合は優先
            if signals['mean_reversion'] != 0 and (signals['trend'] == 0 or signals['breakout'] == 0):
                # Mean Reversion単独または一つの戦略とのみ一致する場合、高い重みを設定
                weights = {'mean_reversion': 0.85, 'trend': 0.075, 'breakout': 0.075}
                
                # 以下の計算をスキップ
                weighted_signal = signals['mean_reversion'] * 0.85
                if signals['trend'] != 0:
                    weighted_signal += signals['trend'] * 0.075
                if signals['breakout'] != 0:
                    weighted_signal += signals['breakout'] * 0.075
                
                # 最終シグナルの判定
                final_signal = 0
                if weighted_signal >= self.config['buy_threshold']:
                    final_signal = 1
                elif weighted_signal <= self.config['sell_threshold']:
                    final_signal = -1
                
                # 理由文字列を生成
                signal_reason = self._generate_signal_reason(
                    final_signal, signals, weights, is_trending, atr_ratio,
                    mean_reversion_signal.get('signal_reasons', [])
                )
                
                # 統合されたシグナル情報をまとめる
                integrated_info = {
                    'timestamp': current['timestamp'],
                    'open': current['open'],
                    'high': current['high'],
                    'low': current['low'],
                    'close': current['close'],
                    'signal': final_signal,
                    'weighted_signal': weighted_signal,
                    'adx': adx_value,
                    'is_trending': is_trending,
                    'atr_ratio': atr_ratio,
                    'strategy_agreement': 1,  # MR戦略優先
                    'strategy_signals': signals,
                    'strategy_weights': weights,
                    'signal_strengths': signal_strengths,
                    'signal_reason': signal_reason
                }
                
                # Mean Reversionの詳細情報も追加
                if signals['mean_reversion'] != 0:
                    integrated_info['mean_reversion_details'] = {
                        'strength': signal_strengths['mean_reversion'],
                        'reasons': mean_reversion_signal.get('signal_reasons', []),
                        'z_score': mean_reversion_signal.get('z_score', 0)
                    }
                
                # 利用可能な指標情報を追加
                for key in ['RSI', 'MACD', 'BB_upper', 'BB_lower', 'SMA_short', 'SMA_long']:
                    if key in current:
                        integrated_info[key] = current[key]
                
                return integrated_info
        
        # トレンド戦略の品質フィルター
        if self.config['trend_quality_filter'] and signals['trend'] != 0:
            # ADXが低い場合はトレンドシグナルを弱める
            if adx_value < 20:
                signal_strengths['trend'] *= 0.6
            
            # トレンドシグナルの確認要素が少ない場合は弱める
            confirmation_count = trend_signal.get('confirmation_count', 0)
            if confirmation_count < 2:
                signal_strengths['trend'] *= 0.7
        
        # ブレイクアウト戦略の品質フィルター
        if self.config['breakout_quality_filter'] and signals['breakout'] != 0:
            # 重要なレベルでないブレイクアウトは弱める
            if not breakout_signal.get('is_important_level', True):
                signal_strengths['breakout'] *= 0.6
                
            # ボラティリティが低い場合はブレイクアウトシグナルを弱める
            if atr_ratio < 0.008:
                signal_strengths['breakout'] *= 0.7
        
        # 環境とシグナルに基づく戦略の重み付け（キャッシュ機構使用）
        weights = self._calculate_strategy_weights(
            is_trending, is_high_volatility, is_low_volatility, 
            signals, signal_strengths
        )
        
        # 加重平均でシグナルを統合
        weighted_signal = 0
        for strategy, signal in signals.items():
            weighted_signal += signal * weights[strategy]
        
        # 最終シグナルの決定（閾値を適用）
        final_signal = 0
        if weighted_signal >= self.config['buy_threshold']:
            final_signal = 1
        elif weighted_signal <= self.config['sell_threshold']:
            final_signal = -1
        
        # 戦略間の一貫性チェック
        strategy_agreement = sum(1 for s in signals.values() if s > 0) - sum(1 for s in signals.values() if s < 0)
        
        # 平均回帰単独のシグナルも考慮（特別扱い）
        if signals['mean_reversion'] != 0 and signal_strengths['mean_reversion'] > self.config['very_strong_signal_threshold']:
            # 平均回帰が非常に強力な場合、他の戦略からの反対意見があっても考慮
            if weighted_signal * signals['mean_reversion'] > 0:  # 同じ方向
                final_signal = signals['mean_reversion']
        
        # 理由文字列の生成
        signal_reason = self._generate_signal_reason(
            final_signal, signals, weights, is_trending, atr_ratio,
            mean_reversion_signal.get('signal_reasons', [])
        )
        
        # 統合されたシグナル情報をまとめる
        integrated_info = {
            'timestamp': current['timestamp'],
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'signal': final_signal,
            'weighted_signal': weighted_signal,
            'adx': adx_value,
            'is_trending': is_trending,
            'atr_ratio': atr_ratio,
            'strategy_agreement': strategy_agreement,
            'strategy_signals': signals,
            'strategy_weights': weights,
            'signal_strengths': signal_strengths,
            'signal_reason': signal_reason
        }
        
        # Mean Reversionの詳細情報も追加
        if signals['mean_reversion'] != 0:
            integrated_info['mean_reversion_details'] = {
                'strength': signal_strengths['mean_reversion'],
                'reasons': mean_reversion_signal.get('signal_reasons', []),
                'z_score': mean_reversion_signal.get('z_score', 0)
            }
        
        # 利用可能な指標情報を追加
        for key in ['RSI', 'MACD', 'BB_upper', 'BB_lower', 'SMA_short', 'SMA_long']:
            if key in current:
                integrated_info[key] = current[key]
        
        return integrated_info
    
    def _calculate_strategy_weights(self, is_trending, is_high_volatility, is_low_volatility, 
                                  signals, signal_strengths):
        """市場環境に基づく戦略ウェイトの計算（最適化版）"""
        
        # キャッシュ使用（高速化）
        cache_key = (is_trending, is_high_volatility, is_low_volatility)
        
        # 予め計算された共通のシナリオに基づく重みを使用
        if self.config['preserve_mean_reversion']:
            # キャッシュから重みを取得
            if cache_key in self._weight_scenarios_cache:
                weights = self._weight_scenarios_cache[cache_key].copy()
            else:
                # シナリオに基づくウェイトマップを事前定義
                weight_scenarios = {
                    # (is_trending, is_high_vol, is_low_vol): {'trend': w1, 'breakout': w2, 'mean_reversion': w3}
                    (True, True, False): {'trend': 0.45, 'breakout': 0.35, 'mean_reversion': 0.20},
                    (True, False, False): {'trend': 0.50, 'breakout': 0.30, 'mean_reversion': 0.20},
                    (False, False, True): {'trend': 0.05, 'breakout': 0.15, 'mean_reversion': 0.80},
                    (False, False, False): {'trend': 0.15, 'breakout': 0.25, 'mean_reversion': 0.60}
                }
                
                # 現在の市場状況に基づいてウェイトを選択
                if cache_key in weight_scenarios:
                    weights = weight_scenarios[cache_key].copy()
                else:
                    # デフォルトのウェイト
                    weights = {'trend': 0.15, 'breakout': 0.25, 'mean_reversion': 0.60}
                
                # キャッシュに保存
                self._weight_scenarios_cache[cache_key] = weights.copy()
        else:
            # Mean Reversion重みを調整する場合の重み付け
            if is_trending:
                if is_high_volatility:
                    # 高ボラティリティなトレンド
                    weights = {'trend': 0.60, 'breakout': 0.30, 'mean_reversion': 0.10}
                else:
                    # 通常のトレンド
                    weights = {'trend': 0.65, 'breakout': 0.25, 'mean_reversion': 0.10}
            else:
                if is_low_volatility:
                    # 低ボラティリティなレンジ
                    weights = {'trend': 0.05, 'breakout': 0.15, 'mean_reversion': 0.80}
                else:
                    # 通常のレンジ
                    weights = {'trend': 0.15, 'breakout': 0.25, 'mean_reversion': 0.60}
        
        # シグナル強度に基づく重み調整（Mean Reversionは影響を受けない）
        # トレンドシグナルの重み調整
        if signals['trend'] != 0:
            trend_quality = signal_strengths['trend']
            if trend_quality > 0.7:  # 高品質シグナル
                # トレンドの重みをブレイクアウトから移動（Mean Reversionに影響なし）
                weights['trend'] += weights['breakout'] * 0.3
                weights['breakout'] *= 0.7
        
        # ブレイクアウトシグナルの重み調整
        if signals['breakout'] != 0:
            breakout_quality = signal_strengths['breakout']
            if breakout_quality > 0.7:  # 高品質シグナル
                # ブレイクアウトの重みをトレンドから移動（Mean Reversionに影響なし）
                weights['breakout'] += weights['trend'] * 0.3
                weights['trend'] *= 0.7
        
        # 平均回帰シグナルが特に強い場合、その重みをさらに増加
        if abs(signals['mean_reversion']) > 0 and signal_strengths['mean_reversion'] > self.config['strong_signal_threshold']:
            # 他の戦略の重みを減らし、平均回帰の重みを増加
            weight_reduction_trend = weights['trend'] * 0.2
            weight_reduction_breakout = weights['breakout'] * 0.2
            
            weights['trend'] -= weight_reduction_trend
            weights['breakout'] -= weight_reduction_breakout
            weights['mean_reversion'] += (weight_reduction_trend + weight_reduction_breakout)
        
        return weights
    
    def _generate_signal_reason(self, final_signal, signals, weights, is_trending, atr_ratio, mr_reasons):
        """シグナル生成理由の文字列を生成（最適化版）"""
        if final_signal == 0:
            return "シグナルなし"
        
        # 主要戦略の特定
        dominant_strategy = max(weights.items(), key=lambda x: x[1] if signals[x[0]] != 0 else 0)[0]
        
        # 基本的な理由
        signal_type = "買い" if final_signal > 0 else "売り"
        reason = f"{signal_type}シグナル: {dominant_strategy}戦略優勢"
        
        # 市場環境の追加
        env_type = "トレンド相場" if is_trending else "レンジ相場"
        vol_type = ""
        if atr_ratio > 0.015:
            vol_type = "高ボラティリティ"
        elif atr_ratio < 0.006:
            vol_type = "低ボラティリティ"
        
        if vol_type:
            env_type += f"({vol_type})"
        
        reason += f", {env_type}"
        
        # 戦略固有の理由を追加
        if dominant_strategy == 'mean_reversion' and mr_reasons:
            reason += f", 理由: {', '.join(mr_reasons[:2])}"  # 最初の2つの理由を表示
        elif dominant_strategy == 'trend':
            # Trend戦略の改善に関する情報
            if 'confirmation_count' in signals:
                reason += f", 確認指標: {signals.get('confirmation_count', 0)}"
            if is_trending:
                reason += f", ADX: {signals.get('adx', 0):.1f}"
        elif dominant_strategy == 'breakout':
            # Breakout戦略の改善に関する情報
            if signals.get('is_important_level', True):
                reason += ", 重要なレベル"
            else:
                reason += ", 一般的なレベル"
        
        return reason
    
    def _calculate_adx(self, data, period=14):
        """ADX（平均方向性指数）の計算（キャッシュ機能付き）"""
        # キャッシュから取得を試みる
        cache_key = len(data)
        if cache_key in self._adx_cache:
            return self._adx_cache[cache_key]
            
        # データフレームに既に計算済みなら使用
        if 'adx' in data.columns and not pd.isna(data['adx'].iloc[-1]):
            adx_value = data['adx'].iloc[-1]
            # キャッシュに保存
            self._adx_cache[cache_key] = adx_value
            return adx_value
            
        # ADXの列名が大文字の場合もチェック
        if 'ADX' in data.columns and not pd.isna(data['ADX'].iloc[-1]):
            adx_value = data['ADX'].iloc[-1]
            # キャッシュに保存
            self._adx_cache[cache_key] = adx_value
            return adx_value
            
        # 最適化: 簡易ADX計算に切り替え
        # (計算コストの高いADX計算を避けるための簡易版)
        high = data['high'].tail(period * 2).values
        low = data['low'].tail(period * 2).values
        close = data['close'].tail(period * 2).values
        
        # 単純化されたATR計算（正確性よりも速度優先）
        tr = np.zeros(len(high))
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.mean(tr[-period:])
        
        # 方向性指標の簡易計算
        direction = np.sign(close[-1] - close[-period])
        slope = abs(close[-1] - close[-period]) / close[-period]
        
        # 簡易ADX値（0-100の範囲）
        adx_value = min(100, max(0, slope * 100 * (1 + atr / close[-1])))
        
        # キャッシュに保存
        self._adx_cache[cache_key] = adx_value
        
        return adx_value
    
    def adaptive_risk_reward(self, signal_info, base_sl_percent=1.5, base_tp_percent=5.0):
        """
        市場環境とシグナルタイプに基づいて動的にリスク/リワード比を調整（最適化版）
        
        Parameters:
        -----------
        signal_info : dict
            統合されたシグナル情報
        base_sl_percent : float
            基本のストップロス割合（％）
        base_tp_percent : float
            基本のテイクプロフィット割合（％）
            
        Returns:
        --------
        tuple
            (stop_loss_percent, take_profit_percent)
        """
        # 基本のリスク/リワード設定
        sl_percent = base_sl_percent
        tp_percent = base_tp_percent
        
        # どの戦略がシグナルを出したか特定
        dominant_strategy = None
        max_weight = 0
        
        for strategy, weight in signal_info['strategy_weights'].items():
            strategy_signal = signal_info['strategy_signals'][strategy]
            if strategy_signal != 0 and weight > max_weight:
                max_weight = weight
                dominant_strategy = strategy
        
        if dominant_strategy is None:
            return sl_percent, tp_percent
            
        # 戦略タイプに応じた調整
        if dominant_strategy == 'trend':
            # 改善されたトレンドフォロー戦略では早期利益確定を重視
            # TP/SL比率をより最適化
            tp_percent *= 1.2
            
            # ADX値に基づく調整
            adx_value = signal_info.get('adx', 25)
            if adx_value > 35:  # 非常に強いトレンド
                # 強いトレンドではより広いTPとSL
                tp_percent *= 1.2
                sl_percent *= 0.9  # より狭いSLで早期損切り
            
            # RSIに基づく調整
            rsi = signal_info.get('RSI', 50)
            if (signal_info['signal'] > 0 and rsi > 65) or (signal_info['signal'] < 0 and rsi < 35):
                # 極端なRSIではリバーサルの可能性を考慮
                tp_percent *= 0.8  # 早めの利益確定
        
        elif dominant_strategy == 'breakout':
            # 改善されたブレイクアウト戦略では、より広いSLと適切なTP
            tp_percent *= 1.1
            sl_percent *= 1.1  # ブレイクアウト後の揺り戻しを考慮
            
            # 重要なレベルでのブレイクアウトはより大きな値幅を狙える
            if signal_info.get('is_important_level', False):
                tp_percent *= 1.2
            
            # 高ボラティリティ環境ではより広いSL
            if signal_info.get('atr_ratio', 0.01) > 0.015:
                sl_percent *= 1.2
                tp_percent *= 1.1
        
        elif dominant_strategy == 'mean_reversion':
            # 平均回帰戦略の最適化 - 保有期間短縮のため早めのTP
            signal_strength = signal_info.get('signal_strengths', {}).get('mean_reversion', 0.5)
            
            # 早期利益確定を強化
            mr_tp_factor = self.config.get('mr_tp_factor', 0.65)
            tp_percent = tp_percent * mr_tp_factor  # 早めのTeach Profit
            sl_percent = sl_percent * 0.75
        
        # 市場環境（トレンド/レンジ）に基づく調整
        if signal_info.get('is_trending', False):
            # トレンド相場での平均回帰は注意
            if dominant_strategy == 'mean_reversion':
                # トレンド中の平均回帰ではより狭いTP
                tp_percent *= 0.8
        else:
            # レンジ相場での平均回帰は有利
            if dominant_strategy == 'mean_reversion':
                # レンジ相場では平均回帰のSLを狭く
                sl_percent *= 0.85
        
        # ボラティリティに基づく調整
        atr_ratio = signal_info.get('atr_ratio', 0.01)
        
        # 平均回帰は低〜中程度のボラティリティで最適化
        if dominant_strategy == 'mean_reversion':
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
        
        # Mean Reversionの場合、Z-scoreに基づく調整
        if dominant_strategy == 'mean_reversion' and 'mean_reversion_details' in signal_info:
            z_score = abs(signal_info['mean_reversion_details'].get('z_score', 0))
            if z_score > 2.5:  # 極端な乖離
                # より狭いTP（早期に利益確定）
                tp_percent *= 0.8
        
        # 最終値を妥当な範囲に収める
        sl_percent = max(0.8, min(sl_percent, 2.5))
        tp_percent = max(1.5, min(tp_percent, 10.0))
        
        # 平均回帰では最低リスク/リワード比が低くても許容
        if dominant_strategy == 'mean_reversion':
            min_risk_reward = 1.5  # 通常の2.0より低い
        else:
            min_risk_reward = 2.0
        
        # 最低リスク/リワード比の保証
        if tp_percent / sl_percent < min_risk_reward:
            tp_percent = sl_percent * min_risk_reward
        
        return sl_percent, tp_percent