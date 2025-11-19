import os
import pandas as pd
import numpy as np
import joblib
# import lightgbm as lgb # libomp dependency issue
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from loguru import logger
from enhanced_trading_bot import EnhancedTradingBot

# ログ設定
logger.add("logs/ml_trainer_{time}.log", rotation="1 day")

class MLTrainer:
    def __init__(self):
        self.bot = EnhancedTradingBot()
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def prepare_data(self):
        """学習用データの準備"""
        logger.info(f"Fetching data for {self.bot.symbol}...")
        
        # 環境変数から期間を取得、なければデフォルト
        start_time = os.getenv("START_TIME", "2024-01-01T00:00:00")
        end_time = os.getenv("END_TIME", "2025-04-03T00:00:00")
        
        df = self.bot.get_historical_data(start_time=start_time, end_time=end_time, is_backtest=True)
        
        # インジケーター計算（Botのロジックを再利用）
        # キャッシュに新しいインジケーターが含まれていない可能性があるため、CCIでチェック
        if 'CCI' not in df.columns:
            logger.info("Calculating indicators...")
            df = self.bot.calculate_indicators(df)
            
        return df

    def generate_labels(self, df, tp_percent=2.0, sl_percent=1.0, horizon=24):
        """
        正解ラベルの生成
        
        Parameters:
        -----------
        df : pd.DataFrame
            データフレーム
        tp_percent : float
            利確目標（%）
        sl_percent : float
            損切り目標（%）
        horizon : int
            最大保有期間（足数）
            
        Returns:
        --------
        pd.DataFrame
            ラベル付きデータフレーム
        """
        df = df.copy()
        labels = []
        
        # 将来の価格変動を確認
        for i in range(len(df) - horizon):
            current_close = df.iloc[i]['close']
            future_window = df.iloc[i+1 : i+1+horizon]
            
            # 目標価格
            tp_price = current_close * (1 + tp_percent / 100)
            sl_price = current_close * (1 - sl_percent / 100)
            
            # TP/SL到達判定
            hit_tp = False
            hit_sl = False
            
            for _, row in future_window.iterrows():
                if row['high'] >= tp_price:
                    hit_tp = True
                    break
                if row['low'] <= sl_price:
                    hit_sl = True
                    break
            
            # ラベル付け: TP到達かつSL未到達なら1、それ以外は0
            if hit_tp and not hit_sl:
                labels.append(1)
            else:
                labels.append(0)
                
        # 残りのデータはラベルなし（NaN）
        labels.extend([np.nan] * horizon)
        df['label'] = labels
        
        return df.dropna(subset=['label'])

    def extract_features(self, df):
        """特徴量抽出"""
        features = [
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'ADX', 'ATR',
            'CCI', 'MFI', 'williams_r', 'stoch_k', 'stoch_d',
            'volume', 'close'
        ]
        
        # 移動平均乖離率
        df['ma_divergence'] = (df['close'] - df['EMA_short']) / df['EMA_short']
        features.append('ma_divergence')
        
        # ボラティリティ比率
        df['atr_ratio'] = df['ATR'] / df['close']
        features.append('atr_ratio')
        
        # 変化率
        df['roc'] = df['close'].pct_change(periods=1)
        features.append('roc')
        
        # 欠損値処理
        df_clean = df.dropna(subset=features)
        
        return df_clean, features

    def train_model(self, df, features):
        """モデルの学習"""
        X = df[features]
        y = df['label']
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        logger.info(f"Training data size: {len(X_train)}")
        logger.info(f"Test data size: {len(X_test)}")
        logger.info(f"Positive samples ratio: {y.mean():.2%}")
        
        # HistGradientBoostingClassifier (LightGBM like)
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # 評価
        y_pred = model.predict(X_test)
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
        
        return model

    def save_model(self, model, features):
        """モデルと特徴量リストの保存"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"lgbm_model_{timestamp}.pkl")
        features_path = os.path.join(self.model_dir, f"features_{timestamp}.pkl")
        
        # 最新モデルへのシンボリックリンク用パス
        latest_model_path = os.path.join(self.model_dir, "lgbm_model_latest.pkl")
        latest_features_path = os.path.join(self.model_dir, "features_latest.pkl")
        
        joblib.dump(model, model_path)
        joblib.dump(features, features_path)
        
        # 最新版としてコピー（または上書き）
        joblib.dump(model, latest_model_path)
        joblib.dump(features, latest_features_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Features saved to {features_path}")

    def run(self):
        """実行フロー"""
        # 1. データ取得
        df = self.prepare_data()
        
        # 2. ラベル生成
        # 現在の戦略（TP 15%, SL 0.5%）に合わせてラベル生成
        # ただし、ML学習用には少し緩めの条件で「良いエントリー」を学習させるのが一般的
        # ここでは、リスクリワード1:2程度（SL 1%, TP 2%）で「勝ち」を定義してみる
        # または、現在の戦略のTP/SLを使うか...
        # 現在の戦略はTPが遠すぎる(15%)ので、MLで予測するのは難しいかもしれない。
        # 「短期間で価格が上昇するか」を予測対象にする。
        df = self.generate_labels(df, tp_percent=2.0, sl_percent=1.0, horizon=24)
        
        # 3. 特徴量抽出
        df, features = self.extract_features(df)
        
        # 4. 学習
        model = self.train_model(df, features)
        
        # 5. 保存
        self.save_model(model, features)

if __name__ == "__main__":
    trainer = MLTrainer()
    trainer.run()
