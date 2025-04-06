# -*- coding: utf-8 -*-
"""
戦略統合モジュール

複数の取引戦略からのシグナルを統合し、最終的な取引判断を行います。
Mean Reversion戦略の影響力を強化するように最適化されています。
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
            'buy_threshold': 0.25,     # 買いシグナル閾値 (旧: 0.3)
            'sell_threshold': -0.25,   # 売りシグナル閾値 (旧: -0.3)
            'adx_threshold': 25,       # トレンド判定のADX閾値
            'high_vol_threshold': 0.012, # 高ボラティリティ閾値
            'low_vol_threshold': 0.006,  # 低ボラティリティ閾値
            'strong_signal_threshold': 0.7, # 強いシグナル閾値
            'very_strong_signal_threshold': 0.8, # 非常に強いシグナル閾値
        }
        
        # 設定の上書き
        if config:
            self.config.update(config)
    
    def integrate_strategies(self, trend_signal, breakout_signal, mean_reversion_signal, data):
        """
        複数の取引戦略からシグナルを統合
        
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
            'trend': 0.5,  # デフォルト値
            'breakout': 0.5,  # デフォルト値
            'mean_reversion': mean_reversion_signal.get('signal_strength', 0.5)  # 強化版の強度値
        }
        
        # 市場環境の分析（トレンド/レンジ判定）
        adx_value = self._calculate_adx(data)
        is_trending = adx_value > self.config['adx_threshold']
        
        # ボラティリティ評価
        atr = current.get('ATR', current['close'] * 0.01)
        atr_ratio = atr / current['close']
        is_high_volatility = atr_ratio > self.config['high_vol_threshold']
        is_low_volatility = atr_ratio < self.config['low_vol_threshold']
        
        # 環境とシグナルに基づく戦略の重み付け
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
        """市場環境に基づく戦略ウェイトの計算"""
        weights = {
            'trend': 0.0,
            'breakout': 0.0,
            'mean_reversion': 0.0
        }
        
        # トレンド環境での重み付け - mean_reversion の重みを増加
        if is_trending:
            if is_high_volatility:
                # 高ボラティリティなトレンド
                weights['trend'] = 0.45
                weights['breakout'] = 0.35
                weights['mean_reversion'] = 0.20  # 0.1から0.2に増加
            else:
                # 通常のトレンド
                weights['trend'] = 0.50
                weights['breakout'] = 0.30
                weights['mean_reversion'] = 0.20  # 0.1から0.2に増加
        
        # レンジ環境での重み付け - mean_reversion の重みをさらに増加
        else:
            if is_low_volatility:
                # 低ボラティリティなレンジ
                weights['trend'] = 0.05  # 0.1から減少
                weights['breakout'] = 0.15  # 0.2から減少
                weights['mean_reversion'] = 0.80  # 0.7から増加
            else:
                # 通常のレンジ
                weights['trend'] = 0.15  # 0.2から減少
                weights['breakout'] = 0.25  # 0.3から減少
                weights['mean_reversion'] = 0.60  # 0.5から増加
        
        # 平均回帰シグナルが特に強い場合、その重みをさらに増加
        if abs(signals['mean_reversion']) > 0 and signal_strengths['mean_reversion'] > self.config['strong_signal_threshold']:
            # 他の戦略の重みを減らし、平均回帰の重みを増加
            for strategy in ['trend', 'breakout']:
                weights[strategy] *= 0.8  # 20%減少
            
            # 減少分を平均回帰の重みに加算
            weight_reduction = (weights['trend'] * 0.2 + weights['breakout'] * 0.2)
            weights['mean_reversion'] += weight_reduction
        
        return weights
    
    def _generate_signal_reason(self, final_signal, signals, weights, is_trending, atr_ratio, mr_reasons):
        """シグナル生成理由の文字列を生成"""
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
        
        # Mean Reversionの詳細理由を追加
        if dominant_strategy == 'mean_reversion' and mr_reasons:
            reason += f", 理由: {', '.join(mr_reasons[:2])}"  # 最初の2つの理由を表示
        
        return reason
    
    def _calculate_adx(self, data, period=14):
        """ADX（平均方向性指数）の計算"""
        df = data.copy()
        if 'adx' in df.columns and not pd.isna(df['adx'].iloc[-1]):
            return df['adx'].iloc[-1]
            
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
    
    def adaptive_risk_reward(self, signal_info, base_sl_percent=1.5, base_tp_percent=5.0):
        """
        市場環境とシグナルタイプに基づいて動的にリスク/リワード比を調整
        
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
            # トレンドフォロー戦略ではより広いTP設定
            tp_percent *= 1.2
        elif dominant_strategy == 'breakout':
            # ブレイクアウト戦略では迅速なTP/SL
            tp_percent *= 1.1
            sl_percent *= 0.9
        elif dominant_strategy == 'mean_reversion':
            # 平均回帰戦略特有の設定
            signal_strength = signal_info.get('signal_strengths', {}).get('mean_reversion', 0.5)
            
            if signal_strength > self.config['strong_signal_threshold']:  # 強い平均回帰シグナル
                # 強いシグナルではより狭いTP（早めの利確）と狭いSL（小さなリスク）
                tp_percent = tp_percent * 0.7
                sl_percent = sl_percent * 0.7
            else:
                # 通常の平均回帰シグナル
                tp_percent = tp_percent * 0.8
                sl_percent = sl_percent * 0.8
        
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