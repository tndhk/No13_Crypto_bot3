# -*- coding: utf-8 -*-
"""
リスク管理モジュール

トレード勝率を向上させるための包括的なリスク管理機能を提供します。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger


class RiskManager:
    """リスク管理を担当するクラス"""

    def __init__(self, config: Dict = None):
        """
        リスク管理クラスの初期化

        Parameters:
        -----------
        config : dict, optional
            リスク管理設定
        """
        self.config = {
            # ドローダウン制限
            'max_drawdown_percent': 10.0,       # 最大許容ドローダウン（%）
            'daily_loss_limit_percent': 3.0,    # 日次損失制限（%）

            # 連続損失保護
            'max_consecutive_losses': 3,        # 連続損失上限
            'loss_reduction_factor': 0.5,       # 連続損失後のポジションサイズ縮小率

            # トレーリングストップ
            'trailing_stop_activation': 1.5,    # 利益%でトレーリングストップを発動
            'trailing_stop_distance': 0.8,      # トレーリングストップの距離（%）

            # 部分利確
            'partial_take_profit_1': 1.5,       # 第1利確ポイント（%）
            'partial_take_profit_1_ratio': 0.3, # 第1利確比率
            'partial_take_profit_2': 3.0,       # 第2利確ポイント（%）
            'partial_take_profit_2_ratio': 0.3, # 第2利確比率

            # ポジションサイジング
            'max_position_size_percent': 5.0,   # 最大ポジションサイズ（資本の%）
            'volatility_position_scaling': True, # ボラティリティに基づくサイジング

            # 市場フィルター
            'min_adx_for_trend': 25,            # トレンド判定のADX最低値
            'max_adx_for_mean_reversion': 20,   # MR戦略の最大ADX
            'min_volume_ratio': 0.5,            # 最小出来高比率（平均比）
        }

        if config:
            self.config.update(config)

        # 状態管理
        self.peak_balance = 0
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.last_trade_date = None
        self.trailing_stop_price = 0
        self.partial_exits_done = []

    def check_drawdown_limit(self, current_balance: float, initial_balance: float) -> Tuple[bool, str]:
        """
        ドローダウン制限をチェック

        Returns:
        --------
        tuple
            (取引可能かどうか, 理由)
        """
        # ピーク残高を更新
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # ドローダウン計算
        if self.peak_balance > 0:
            drawdown_percent = (self.peak_balance - current_balance) / self.peak_balance * 100
        else:
            drawdown_percent = 0

        if drawdown_percent >= self.config['max_drawdown_percent']:
            return False, f"最大ドローダウン制限到達 ({drawdown_percent:.2f}% >= {self.config['max_drawdown_percent']}%)"

        return True, ""

    def check_daily_loss_limit(self, current_balance: float, day_start_balance: float) -> Tuple[bool, str]:
        """
        日次損失制限をチェック

        Returns:
        --------
        tuple
            (取引可能かどうか, 理由)
        """
        daily_loss_percent = (day_start_balance - current_balance) / day_start_balance * 100

        if daily_loss_percent >= self.config['daily_loss_limit_percent']:
            return False, f"日次損失制限到達 ({daily_loss_percent:.2f}%)"

        return True, ""

    def check_consecutive_losses(self) -> Tuple[bool, float]:
        """
        連続損失をチェックし、ポジションサイズ調整率を返す

        Returns:
        --------
        tuple
            (取引可能かどうか, ポジションサイズ調整率)
        """
        if self.config['max_consecutive_losses'] > 0 and self.consecutive_losses >= self.config['max_consecutive_losses']:
            return False, 0

        # 連続損失に応じてポジションサイズを縮小
        if self.consecutive_losses > 0:
            reduction = self.config['loss_reduction_factor'] ** self.consecutive_losses
            return True, max(reduction, 0.25)  # 最低25%

        return True, 1.0

    def record_trade_result(self, is_win: bool):
        """取引結果を記録"""
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.config['max_consecutive_losses'] > 0 and self.consecutive_losses >= self.config['max_consecutive_losses']:
                logger.warning(f"連続損失が{self.consecutive_losses}回に達しました。取引を一時停止します。")

    def calculate_position_size(self, balance: float, atr: float, current_price: float) -> float:
        """
        ATRベースのポジションサイジング
        
        Parameters:
        -----------
        atr : float
            ATR値
        current_price : float
            現在価格
            
        Returns:
        --------
        float
            計算されたポジションサイズ（数量）
        """
        if atr <= 0 or current_price <= 0:
            return self.config.get('base_quantity', 0.025)
            
        # 1トレードあたりの許容リスク額（残高のN%）
        # configにない場合はデフォルト2.0% (調整: 1.0 -> 2.0)
        risk_percent = self.config.get('risk_per_trade_percent', 2.0)
        risk_per_trade = balance * (risk_percent / 100.0)
        
        # ストップロス幅（ATRの倍数）
        # 実際のSLが0.5%とタイトなため、計算上の幅も狭めてサイズを確保する
        stop_loss_distance = atr * 1.0 # 調整: 2.0 -> 1.0
        
        # ポジションサイズ計算: リスク額 / ストップ幅
        # Risk = Size * StopDistance  =>  Size = Risk / StopDistance
        position_size = risk_per_trade / stop_loss_distance
        
        # 最小・最大サイズの制限
        min_size = 0.001 # BTCの最小単位（例）
        max_size = (balance * 0.95) / current_price # 全力買いの95%まで
        
        # configの最大ポジションサイズ制限も考慮
        if 'max_position_size_percent' in self.config:
            max_allowed = (balance * self.config['max_position_size_percent'] / 100.0) / current_price
            max_size = min(max_size, max_allowed)
        
        position_size = max(min_size, min(position_size, max_size))
        
        return position_size

        # 連続損失による縮小
        can_trade, size_factor = self.check_consecutive_losses()
        if not can_trade:
            return 0

        return position_size * size_factor

    def calculate_trailing_stop(self, entry_price: float, current_price: float,
                                 is_long: bool = True) -> Optional[float]:
        """
        トレーリングストップ価格を計算

        Parameters:
        -----------
        entry_price : float
            エントリー価格
        current_price : float
            現在価格
        is_long : bool
            ロングポジションかどうか

        Returns:
        --------
        float or None
            トレーリングストップ価格（発動しない場合はNone）
        """
        if is_long:
            profit_percent = (current_price - entry_price) / entry_price * 100

            # アクティベーションレベルに達したら
            if profit_percent >= self.config['trailing_stop_activation']:
                new_stop = current_price * (1 - self.config['trailing_stop_distance'] / 100)

                # ストップ価格は上昇のみ
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop

                return self.trailing_stop_price
        else:
            # ショートポジションの場合（現状未使用）
            pass

        return None

    def check_partial_take_profit(self, entry_price: float, current_price: float,
                                   quantity: float) -> Tuple[float, float]:
        """
        部分利確をチェック

        Parameters:
        -----------
        entry_price : float
            エントリー価格
        current_price : float
            現在価格
        quantity : float
            現在のポジション量

        Returns:
        --------
        tuple
            (利確する量, 残りの量)
        """
        profit_percent = (current_price - entry_price) / entry_price * 100
        take_quantity = 0

        # 第1利確
        if profit_percent >= self.config['partial_take_profit_1'] and 1 not in self.partial_exits_done:
            take_quantity = quantity * self.config['partial_take_profit_1_ratio']
            self.partial_exits_done.append(1)
            logger.info(f"第1部分利確: {profit_percent:.2f}% で {self.config['partial_take_profit_1_ratio']*100}% 決済")

        # 第2利確
        elif profit_percent >= self.config['partial_take_profit_2'] and 2 not in self.partial_exits_done:
            take_quantity = quantity * self.config['partial_take_profit_2_ratio']
            self.partial_exits_done.append(2)
            logger.info(f"第2部分利確: {profit_percent:.2f}% で {self.config['partial_take_profit_2_ratio']*100}% 決済")

        remaining = quantity - take_quantity
        return take_quantity, remaining

    def reset_position_state(self):
        """ポジション関連の状態をリセット"""
        self.trailing_stop_price = 0
        self.partial_exits_done = []

    def should_skip_trade(self, signal_info: Dict, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        市場環境に基づいてトレードをスキップすべきか判断

        Parameters:
        -----------
        signal_info : dict
            シグナル情報
        data : pd.DataFrame
            市場データ

        Returns:
        --------
        tuple
            (スキップすべきか, 理由)
        """
        current = data.iloc[-1]

        # ADXチェック
        adx = signal_info.get('adx', 0)
        strategy = self._get_dominant_strategy(signal_info)

        # Mean Reversionはトレンド相場を避ける
        if strategy == 'mean_reversion' and adx > self.config['max_adx_for_mean_reversion']:
            return True, f"Mean Reversion: ADX({adx:.1f})が高すぎる（トレンド相場）"

        # トレンド戦略はレンジ相場を避ける
        if strategy == 'trend' and adx < self.config['min_adx_for_trend']:
            return True, f"Trend: ADX({adx:.1f})が低すぎる（レンジ相場）"

        # 出来高チェック
        if 'volume' in current and 'volume_ma' in current:
            volume_ratio = current['volume'] / current['volume_ma']
            if volume_ratio < self.config['min_volume_ratio']:
                return True, f"出来高不足（平均の{volume_ratio*100:.0f}%）"

        # 極端なボラティリティチェック
        atr_ratio = signal_info.get('atr_ratio', 0)
        if strategy == 'mean_reversion' and atr_ratio > 0.025:
            return True, f"Mean Reversion: ボラティリティが高すぎる({atr_ratio*100:.2f}%)"

        return False, ""

    def _get_dominant_strategy(self, signal_info: Dict) -> str:
        """シグナル情報から主要戦略を取得"""
        strategy_weights = signal_info.get('strategy_weights', {})
        strategy_signals = signal_info.get('strategy_signals', {})

        dominant = 'unknown'
        max_weight = 0

        for strategy, weight in strategy_weights.items():
            if strategy_signals.get(strategy, 0) != 0 and weight > max_weight:
                max_weight = weight
                dominant = strategy

        return dominant

    def calculate_dynamic_stop_loss(self, entry_price: float, atr: float,
                                     signal_strength: float) -> float:
        """
        動的ストップロスを計算

        Parameters:
        -----------
        entry_price : float
            エントリー価格
        atr : float
            ATR値
        signal_strength : float
            シグナル強度

        Returns:
        --------
        float
            ストップロス価格
        """
        # 基本は2ATR
        base_multiplier = 2.0

        # シグナル強度に応じて調整
        if signal_strength > 0.8:
            multiplier = base_multiplier * 0.8  # 強いシグナルは狭いSL
        elif signal_strength < 0.5:
            multiplier = base_multiplier * 1.2  # 弱いシグナルは広いSL
        else:
            multiplier = base_multiplier

        stop_distance = atr * multiplier
        stop_loss = entry_price - stop_distance

        return stop_loss

    def calculate_dynamic_take_profit(self, entry_price: float, atr: float,
                                       signal_strength: float) -> float:
        """
        動的テイクプロフィットを計算

        Parameters:
        -----------
        entry_price : float
            エントリー価格
        atr : float
            ATR値
        signal_strength : float
            シグナル強度

        Returns:
        --------
        float
            テイクプロフィット価格
        """
        # 基本は3ATR（R:R = 1.5）
        base_multiplier = 3.0

        # シグナル強度に応じて調整
        if signal_strength > 0.8:
            multiplier = base_multiplier * 1.3  # 強いシグナルは広いTP
        elif signal_strength < 0.5:
            multiplier = base_multiplier * 0.8  # 弱いシグナルは狭いTP
        else:
            multiplier = base_multiplier

        profit_distance = atr * multiplier
        take_profit = entry_price + profit_distance

        return take_profit

    def get_status_report(self) -> Dict:
        """リスク管理の状態レポートを取得"""
        return {
            'peak_balance': self.peak_balance,
            'consecutive_losses': self.consecutive_losses,
            'trailing_stop_price': self.trailing_stop_price,
            'partial_exits_done': self.partial_exits_done,
        }

    def detect_market_regime(self, data: pd.DataFrame) -> Dict:
        """
        市場レジームを検出

        Parameters:
        -----------
        data : pd.DataFrame
            市場データ

        Returns:
        --------
        dict
            市場レジーム情報
        """
        if len(data) < 50:
            return {'regime': 'unknown', 'confidence': 0}

        current = data.iloc[-1]

        # 1. トレンド方向の判定
        close_series = data['close'].tail(50)
        sma_20 = close_series.rolling(20).mean().iloc[-1]
        sma_50 = close_series.rolling(50).mean().iloc[-1]

        if current['close'] > sma_20 > sma_50:
            trend_direction = 'bullish'
        elif current['close'] < sma_20 < sma_50:
            trend_direction = 'bearish'
        else:
            trend_direction = 'sideways'

        # 2. ボラティリティレジーム
        atr_ratio = current.get('atr_ratio', 0.01)
        if atr_ratio > 0.02:
            volatility_regime = 'high'
        elif atr_ratio < 0.008:
            volatility_regime = 'low'
        else:
            volatility_regime = 'normal'

        # 3. モメンタム
        rsi = current.get('RSI', 50)
        if rsi > 70:
            momentum = 'overbought'
        elif rsi < 30:
            momentum = 'oversold'
        else:
            momentum = 'neutral'

        # 4. 市場フェーズ（accumulation/distribution）
        volume = current.get('volume', 0)
        volume_ma = current.get('volume_ma', volume)
        if volume_ma > 0:
            volume_ratio = volume / volume_ma
            if volume_ratio > 1.5 and trend_direction == 'bullish':
                market_phase = 'distribution'
            elif volume_ratio > 1.5 and trend_direction == 'bearish':
                market_phase = 'accumulation'
            else:
                market_phase = 'neutral'
        else:
            market_phase = 'neutral'
            
        return {
            'trend_direction': trend_direction,
            'volatility_regime': volatility_regime,
            'momentum': momentum,
            'market_phase': market_phase
        }

    def should_skip_trade(self, signal_info: Dict, prev_data: pd.DataFrame) -> Tuple[bool, str]:
        """
        リスク要因に基づいて取引をスキップすべきか判定
        
        Parameters:
        -----------
        signal_info : dict
            シグナル情報（atr_ratio, atr_changeなどを含む）
        prev_data : pd.DataFrame
            過去データ
            
        Returns:
        --------
        tuple
            (スキップすべきか, 理由)
        """
        # ボラティリティフィルターはパフォーマンス低下のため無効化
        return False, ""
        
        # 1. 極端なボラティリティ拡大（フラッシュクラッシュ/急騰）
        # atr_change = signal_info.get('atr_change', 0)
        # if atr_change > 0.5: # ATRが50%以上急拡大 (緩和: 30% -> 50%)
        #     return True, f"ボラティリティ急拡大(ATR変化率: {atr_change:.2%})"
            
        # 2. 極端な高ボラティリティ
        # atr_ratio = signal_info.get('atr_ratio', 0)
        # if atr_ratio > 0.06: # 価格の6%以上の変動幅 (緩和: 4% -> 6%)
        #     return True, f"極端な高ボラティリティ(ATR比率: {atr_ratio:.2%})"
            
        # 3. 極端な低ボラティリティ（動きがない）
        # if atr_ratio < 0.001: # 価格の0.1%未満の変動幅 (緩和: 0.2% -> 0.1%)
        #     return True, f"極端な低ボラティリティ(ATR比率: {atr_ratio:.2%})"
            
        # return False, ""

    def advanced_market_filter(self, signal_info: Dict, prev_data: pd.DataFrame) -> Tuple[bool, str]:
        """高度な市場フィルター（将来的な拡張用）"""
        return False, ""

    def calculate_optimal_entry_timing(self, signal_info: Dict, prev_data: pd.DataFrame) -> Dict:
        """エントリータイミングの最適化（簡易版）"""
        return {'entry_quality': 'good', 'score': 5}

        # 5. ADXによるトレンド強度
        adx = current.get('ADX', 0)
        if adx > 40:
            trend_strength = 'strong'
        elif adx > 25:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'

        return {
            'trend_direction': trend_direction,
            'volatility_regime': volatility_regime,
            'momentum': momentum,
            'market_phase': market_phase,
            'trend_strength': trend_strength,
            'adx': adx,
            'rsi': rsi,
            'atr_ratio': atr_ratio
        }

    def advanced_market_filter(self, signal_info: Dict, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        高度な市場環境フィルター

        Parameters:
        -----------
        signal_info : dict
            シグナル情報
        data : pd.DataFrame
            市場データ

        Returns:
        --------
        tuple
            (スキップすべきか, 理由)
        """
        if len(data) < 50:
            return False, ""

        regime = self.detect_market_regime(data)
        strategy = self._get_dominant_strategy(signal_info)

        # Mean Reversion戦略のフィルター
        if strategy == 'mean_reversion':
            # 強いトレンド中は避ける
            if regime['trend_strength'] == 'strong' and regime['adx'] > 35:
                return True, f"強いトレンド中 (ADX={regime['adx']:.1f})"

            # 高ボラティリティ環境では注意
            if regime['volatility_regime'] == 'high' and regime['atr_ratio'] > 0.025:
                return True, f"高ボラティリティ ({regime['atr_ratio']*100:.2f}%)"

            # オーバーバウト/オーバーソールドでの逆張り確認
            signal_direction = signal_info.get('signal', 0)
            if signal_direction == 1 and regime['momentum'] == 'overbought':
                return True, "RSIがオーバーバウト状態で買いシグナル"
            if signal_direction == -1 and regime['momentum'] == 'oversold':
                return True, "RSIがオーバーソールド状態で売りシグナル"

        # トレンド戦略のフィルター
        elif strategy == 'trend':
            # レンジ相場では避ける
            if regime['trend_strength'] == 'weak':
                return True, f"弱いトレンド (ADX={regime['adx']:.1f})"

            # 低ボラティリティではトレンドが弱い
            if regime['volatility_regime'] == 'low':
                return True, "低ボラティリティ環境"

        # ブレイクアウト戦略のフィルター
        elif strategy == 'breakout':
            # 出来高が低い時は避ける
            current = data.iloc[-1]
            volume = current.get('volume', 0)
            volume_ma = current.get('volume_ma', volume)
            if volume_ma > 0 and volume / volume_ma < 0.7:
                return True, "出来高不足"

        # 共通フィルター: 極端な市場状態
        if regime['volatility_regime'] == 'high' and regime['momentum'] in ['overbought', 'oversold']:
            return True, "極端な市場状態（高ボラティリティ + 極端なRSI）"

        return False, ""

    def calculate_optimal_entry_timing(self, signal_info: Dict, data: pd.DataFrame) -> Dict:
        """
        最適なエントリータイミングを計算

        Parameters:
        -----------
        signal_info : dict
            シグナル情報
        data : pd.DataFrame
            市場データ

        Returns:
        --------
        dict
            エントリー推奨情報
        """
        if len(data) < 20:
            return {'entry_quality': 'unknown', 'score': 0}

        current = data.iloc[-1]
        score = 0
        factors = []

        # 1. トレンド方向との一致
        signal = signal_info.get('signal', 0)
        sma_short = current.get('SMA_short', current['close'])
        sma_long = current.get('SMA_long', current['close'])

        if signal == 1 and sma_short > sma_long:
            score += 1
            factors.append('トレンド方向一致')
        elif signal == -1 and sma_short < sma_long:
            score += 1
            factors.append('トレンド方向一致')

        # 2. RSIの位置
        rsi = current.get('RSI', 50)
        if signal == 1 and 30 <= rsi <= 50:
            score += 1
            factors.append('RSI適正（買い）')
        elif signal == -1 and 50 <= rsi <= 70:
            score += 1
            factors.append('RSI適正（売り）')

        # 3. ボリンジャーバンドの位置
        close = current['close']
        bb_upper = current.get('BB_upper', close * 1.02)
        bb_lower = current.get('BB_lower', close * 0.98)
        bb_middle = current.get('BB_middle', close)

        if signal == 1 and close < bb_middle:
            score += 1
            factors.append('BB下半分で買い')
        elif signal == -1 and close > bb_middle:
            score += 1
            factors.append('BB上半分で売り')

        # 4. MACDの確認
        macd_hist = current.get('MACD_hist', 0)
        macd_hist_prev = current.get('MACD_hist_prev', 0)

        if signal == 1 and macd_hist > macd_hist_prev:
            score += 1
            factors.append('MACDヒストグラム上昇')
        elif signal == -1 and macd_hist < macd_hist_prev:
            score += 1
            factors.append('MACDヒストグラム下降')

        # 5. 出来高確認
        volume = current.get('volume', 0)
        volume_ma = current.get('volume_ma', volume)
        if volume_ma > 0 and volume / volume_ma > 1.0:
            score += 1
            factors.append('出来高確認')

        # エントリー品質の判定
        if score >= 4:
            entry_quality = 'excellent'
        elif score >= 3:
            entry_quality = 'good'
        elif score >= 2:
            entry_quality = 'fair'
        else:
            entry_quality = 'poor'

        return {
            'entry_quality': entry_quality,
            'score': score,
            'max_score': 5,
            'factors': factors
        }
