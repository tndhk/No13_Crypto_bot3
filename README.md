# 強化版マルチ戦略トレーディングボット

このプロジェクトは、`Mean Reversion`戦略を強化した暗号資産トレーディングボットです。バックテストモードで動作し、高い勝率とより多くの取引回数を実現するように最適化されています。

## 主な特徴

- Mean Reversion戦略の検出感度を高めて取引回数を増加
- 複数の戦略（トレンド、ブレイクアウト、平均回帰）を統合
- 市場環境に応じた動的戦略重み付け
- 詳細なバックテスト分析と可視化

## 必要条件

- Docker と Docker Compose
- または Python 3.10以上と必要なパッケージ

## クイックスタート（Docker使用）

1. リポジトリをクローンして移動
```bash
git clone <repository-url>
cd enhanced-trading-bot
```

2. `.env`ファイルを設定（テンプレートをコピーして編集）
```bash
cp .env.example .env
# .envファイルを必要に応じて編集
```

3. 実行スクリプトを使用して起動
```bash
chmod +x run_docker.sh
./run_docker.sh
```

または Docker Compose を使用
```bash
docker-compose up --build
```

## クイックスタート（Python環境）

1. リポジトリをクローンして移動
```bash
git clone <repository-url>
cd enhanced-trading-bot
```

2. Python環境のセットアップ
```bash
pip install -r requirements.txt
```

3. `.env`ファイルを設定（テンプレートをコピーして編集）

4. バックテストを実行
```bash
python backtest_runner.py --start 2024-01-01T00:00:00 --end 2025-04-03T00:00:00
```

## 設定オプション

`.env`ファイルまたはコマンドライン引数で以下のオプションを設定できます：

### 基本設定
- `SYMBOL`: 取引ペア（例: BTCUSDT）
- `INTERVAL`: 時間足（例: 1h, 4h, 1d）
- `START_TIME`/`END_TIME`: バックテスト期間

### 戦略設定
- `SHORT_WINDOW`/`LONG_WINDOW`: 移動平均の期間
- `STOP_LOSS_PERCENT`/`TAKE_PROFIT_PERCENT`: リスク設定
- `MR_RSI_LOWER`/`MR_RSI_UPPER`: RSIの閾値
- `MR_BB_LOWER_FACTOR`/`MR_BB_UPPER_FACTOR`: ボリンジャーバンド乗数
- `STRATEGY_BUY_THRESHOLD`/`STRATEGY_SELL_THRESHOLD`: シグナル閾値

## プロジェクト構造

```
enhanced_bot/
├── strategies/             # 各戦略のモジュール
│   ├── __init__.py
│   ├── mean_reversion.py   # 強化版平均回帰戦略
│   ├── trend.py            # トレンドフォロー戦略
│   └── breakout.py         # ブレイクアウト戦略
├── strategy_integrator.py  # 戦略統合モジュール
├── enhanced_trading_bot.py # メインボットクラス
├── backtest_runner.py      # バックテスト実行ツール
├── Dockerfile              # Dockerビルド定義
├── docker-compose.yml      # Docker Compose設定
├── requirements.txt        # 依存パッケージ
└── .env                    # 環境変数設定
```

## 結果の解析

バックテスト結果は`results`ディレクトリに保存されます：

- バックテスト概要（JSON）
- 取引詳細（CSV）
- 残高履歴（CSV）
- パフォーマンスグラフ（PNG）
- 戦略別パフォーマンス分析（TXT）

## Dockerコマンド例

特定の期間でバックテストを実行：
```bash
docker run --rm --env-file .env -v $(pwd)/results:/app/results enhanced-trading-bot --start 2024-01-01T00:00:00 --end 2024-03-31T00:00:00
```

異なる取引ペアでバックテストを実行：
```bash
docker run --rm --env-file .env -v $(pwd)/results:/app/results enhanced-trading-bot --symbol ETHUSDT
```

## ライセンス

MIT