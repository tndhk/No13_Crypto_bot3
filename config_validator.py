# -*- coding: utf-8 -*-
"""
設定値バリデーションモジュール

トレーディングボットの設定パラメータを検証し、
エラーや警告を提供します。
"""

from typing import Dict, List, Any
from loguru import logger


class ConfigValidationError(Exception):
    """設定バリデーションエラー"""
    pass


class ConfigValidator:
    """設定値を検証するクラス"""

    # 有効なインターバル値
    VALID_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

    def __init__(self):
        """バリデーターの初期化"""
        self.warnings: List[str] = []

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        設定値を検証

        Parameters:
        -----------
        config : dict
            検証する設定辞書

        Returns:
        --------
        list
            警告メッセージのリスト

        Raises:
        -------
        ConfigValidationError
            設定値が無効な場合
        """
        self.warnings = []

        # 各パラメータの検証
        self._validate_stop_loss(config)
        self._validate_take_profit(config)
        self._validate_rsi(config)
        self._validate_fees(config)
        self._validate_quantity(config)
        self._validate_windows(config)
        self._validate_interval(config)
        self._validate_slippage(config)
        self._validate_z_score(config)
        self._validate_risk_reward_ratio(config)

        return self.warnings

    def _validate_stop_loss(self, config: Dict[str, Any]):
        """ストップロスの検証"""
        if 'stop_loss_percent' in config:
            sl = config['stop_loss_percent']
            if sl < 0:
                raise ConfigValidationError(
                    f"stop_loss_percent は正の値である必要があります: {sl}"
                )
            if sl > 20:
                raise ConfigValidationError(
                    f"stop_loss_percent が大きすぎます (max 20%): {sl}"
                )
            if sl > 10:
                self.warnings.append(
                    f"stop_loss_percent が高い値です ({sl}%)。リスク管理を確認してください。"
                )

    def _validate_take_profit(self, config: Dict[str, Any]):
        """テイクプロフィットの検証"""
        if 'take_profit_percent' in config:
            tp = config['take_profit_percent']
            if tp < 0:
                raise ConfigValidationError(
                    f"take_profit_percent は正の値である必要があります: {tp}"
                )
            if tp > 100:
                raise ConfigValidationError(
                    f"take_profit_percent が大きすぎます (max 100%): {tp}"
                )

    def _validate_rsi(self, config: Dict[str, Any]):
        """RSI閾値の検証"""
        if 'rsi_upper' in config:
            rsi_upper = config['rsi_upper']
            if rsi_upper < 0 or rsi_upper > 100:
                raise ConfigValidationError(
                    f"rsi_upper は 0-100 の範囲である必要があります: {rsi_upper}"
                )

        if 'rsi_lower' in config:
            rsi_lower = config['rsi_lower']
            if rsi_lower < 0 or rsi_lower > 100:
                raise ConfigValidationError(
                    f"rsi_lower は 0-100 の範囲である必要があります: {rsi_lower}"
                )

        # RSI閾値の順序チェック
        if 'rsi_upper' in config and 'rsi_lower' in config:
            if config['rsi_lower'] >= config['rsi_upper']:
                self.warnings.append(
                    f"rsi_lower ({config['rsi_lower']}) が rsi_upper ({config['rsi_upper']}) 以上です"
                )

    def _validate_fees(self, config: Dict[str, Any]):
        """手数料の検証"""
        for fee_name in ['maker_fee', 'taker_fee']:
            if fee_name in config:
                fee = config[fee_name]
                if fee < 0:
                    raise ConfigValidationError(
                        f"{fee_name} は正の値である必要があります: {fee}"
                    )
                if fee > 0.01:  # 1%以上
                    self.warnings.append(
                        f"{fee_name} が高い値です ({fee * 100}%)。確認してください。"
                    )

    def _validate_quantity(self, config: Dict[str, Any]):
        """取引数量の検証"""
        if 'quantity' in config:
            qty = config['quantity']
            if qty <= 0:
                raise ConfigValidationError(
                    f"quantity は正の値である必要があります: {qty}"
                )

    def _validate_windows(self, config: Dict[str, Any]):
        """移動平均ウィンドウの検証"""
        if 'short_window' in config and 'long_window' in config:
            short = config['short_window']
            long = config['long_window']

            if short <= 0 or long <= 0:
                raise ConfigValidationError(
                    f"ウィンドウサイズは正の値である必要があります: short={short}, long={long}"
                )

            if short >= long:
                self.warnings.append(
                    f"short_window ({short}) が long_window ({long}) 以上です。通常は short < long にします。"
                )

    def _validate_interval(self, config: Dict[str, Any]):
        """インターバルの検証"""
        if 'interval' in config:
            interval = config['interval']
            if interval not in self.VALID_INTERVALS:
                raise ConfigValidationError(
                    f"interval が無効です: {interval}。有効な値: {', '.join(self.VALID_INTERVALS)}"
                )

    def _validate_slippage(self, config: Dict[str, Any]):
        """スリッページの検証"""
        if 'slippage_mean' in config:
            slippage = config['slippage_mean']
            if slippage < 0:
                raise ConfigValidationError(
                    f"slippage_mean は正の値である必要があります: {slippage}"
                )
            if slippage > 0.01:  # 1%以上
                raise ConfigValidationError(
                    f"slippage_mean が大きすぎます (max 1%): {slippage}"
                )

        if 'slippage_std' in config:
            slippage_std = config['slippage_std']
            if slippage_std < 0:
                raise ConfigValidationError(
                    f"slippage_std は正の値である必要があります: {slippage_std}"
                )

    def _validate_z_score(self, config: Dict[str, Any]):
        """Z-score閾値の検証"""
        if 'z_score_threshold' in config:
            z_score = config['z_score_threshold']
            if z_score <= 0:
                raise ConfigValidationError(
                    f"z_score_threshold は正の値である必要があります: {z_score}"
                )
            if z_score > 5:
                self.warnings.append(
                    f"z_score_threshold が高い値です ({z_score})。シグナルが少なくなる可能性があります。"
                )

    def _validate_risk_reward_ratio(self, config: Dict[str, Any]):
        """リスク/リワード比の検証"""
        if 'stop_loss_percent' in config and 'take_profit_percent' in config:
            sl = config['stop_loss_percent']
            tp = config['take_profit_percent']

            if sl > 0:
                ratio = tp / sl
                if ratio < 1.0:
                    self.warnings.append(
                        f"リスク/リワード比が低いです ({ratio:.2f})。TP ({tp}%) が SL ({sl}%) より小さいです。"
                    )
                elif ratio < 1.5:
                    self.warnings.append(
                        f"リスク/リワード比が推奨値より低いです ({ratio:.2f})。最低 1.5 を推奨します。"
                    )


def validate_bot_config(config: Dict[str, Any]) -> List[str]:
    """
    ボット設定を検証するヘルパー関数

    Parameters:
    -----------
    config : dict
        検証する設定辞書

    Returns:
    --------
    list
        警告メッセージのリスト

    Raises:
    -------
    ConfigValidationError
        設定値が無効な場合
    """
    validator = ConfigValidator()
    return validator.validate(config)
