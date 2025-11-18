# -*- coding: utf-8 -*-
"""
設定値バリデーションのユニットテスト
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_validator import ConfigValidator, ConfigValidationError


class TestConfigValidator:
    """設定バリデーションのテストクラス"""

    def test_valid_config(self, bot_config):
        """有効な設定がバリデーションを通過することを確認"""
        validator = ConfigValidator()
        # エラーが発生しなければ成功
        validator.validate(bot_config)

    def test_invalid_stop_loss_negative(self):
        """負のストップロスがエラーになることを確認"""
        validator = ConfigValidator()
        config = {'stop_loss_percent': -1.0}

        with pytest.raises(ConfigValidationError) as excinfo:
            validator.validate(config)

        assert 'stop_loss_percent' in str(excinfo.value)

    def test_invalid_stop_loss_too_high(self):
        """高すぎるストップロスがエラーになることを確認"""
        validator = ConfigValidator()
        config = {'stop_loss_percent': 50.0}

        with pytest.raises(ConfigValidationError) as excinfo:
            validator.validate(config)

        assert 'stop_loss_percent' in str(excinfo.value)

    def test_invalid_take_profit_negative(self):
        """負のテイクプロフィットがエラーになることを確認"""
        validator = ConfigValidator()
        config = {'take_profit_percent': -5.0}

        with pytest.raises(ConfigValidationError) as excinfo:
            validator.validate(config)

        assert 'take_profit_percent' in str(excinfo.value)

    def test_invalid_rsi_upper_out_of_range(self):
        """RSI上限が100を超える場合エラーになることを確認"""
        validator = ConfigValidator()
        config = {'rsi_upper': 110}

        with pytest.raises(ConfigValidationError) as excinfo:
            validator.validate(config)

        assert 'rsi_upper' in str(excinfo.value)

    def test_invalid_rsi_lower_negative(self):
        """RSI下限が負の場合エラーになることを確認"""
        validator = ConfigValidator()
        config = {'rsi_lower': -10}

        with pytest.raises(ConfigValidationError) as excinfo:
            validator.validate(config)

        assert 'rsi_lower' in str(excinfo.value)

    def test_invalid_fee_negative(self):
        """負の手数料がエラーになることを確認"""
        validator = ConfigValidator()
        config = {'maker_fee': -0.001}

        with pytest.raises(ConfigValidationError) as excinfo:
            validator.validate(config)

        assert 'maker_fee' in str(excinfo.value)

    def test_invalid_quantity_zero(self):
        """数量が0の場合エラーになることを確認"""
        validator = ConfigValidator()
        config = {'quantity': 0}

        with pytest.raises(ConfigValidationError) as excinfo:
            validator.validate(config)

        assert 'quantity' in str(excinfo.value)

    def test_invalid_window_order(self):
        """短期ウィンドウが長期より大きい場合警告を出すことを確認"""
        validator = ConfigValidator()
        config = {
            'short_window': 20,
            'long_window': 10
        }

        warnings = validator.validate(config)
        assert any('window' in w.lower() for w in warnings)

    def test_risk_reward_ratio_warning(self):
        """リスク/リワード比が低い場合警告を出すことを確認"""
        validator = ConfigValidator()
        config = {
            'stop_loss_percent': 5.0,
            'take_profit_percent': 3.0
        }

        warnings = validator.validate(config)
        assert any('リスク' in w or 'risk' in w.lower() for w in warnings)

    def test_valid_symbol_format(self):
        """有効なシンボル形式を確認"""
        validator = ConfigValidator()
        config = {'symbol': 'BTCUSDT'}

        # エラーが発生しなければ成功
        validator.validate(config)

    def test_invalid_interval(self):
        """無効なインターバルがエラーになることを確認"""
        validator = ConfigValidator()
        config = {'interval': 'invalid'}

        with pytest.raises(ConfigValidationError) as excinfo:
            validator.validate(config)

        assert 'interval' in str(excinfo.value)

    def test_valid_intervals(self):
        """有効なインターバルがバリデーションを通過することを確認"""
        validator = ConfigValidator()

        valid_intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

        for interval in valid_intervals:
            config = {'interval': interval}
            # エラーが発生しなければ成功
            validator.validate(config)

    def test_slippage_reasonable_range(self):
        """スリッページが妥当な範囲内であることを確認"""
        validator = ConfigValidator()
        config = {'slippage_mean': 0.1}  # 10%は高すぎる

        with pytest.raises(ConfigValidationError) as excinfo:
            validator.validate(config)

        assert 'slippage' in str(excinfo.value)

    def test_z_score_threshold_positive(self):
        """Z-score閾値が正であることを確認"""
        validator = ConfigValidator()
        config = {'z_score_threshold': -1.0}

        with pytest.raises(ConfigValidationError) as excinfo:
            validator.validate(config)

        assert 'z_score' in str(excinfo.value)
