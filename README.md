# 強化版マルチ戦略トレーディングボット

このプロジェクトは、`Mean Reversion`戦略を強化した暗号資産トレーディングボットです。バックテストモードで動作し、高い勝率とより多くの取引回数を実現するように最適化されています。

## 主な特徴

- Mean Reversion戦略の検出感度を高めて取引回数を増加
- 複数の戦略（トレンド、ブレイクアウト、平均回帰）を統合
- 市場環境に応じた動的戦略重み付け
- 詳細なバックテスト分析と可視化
- **包括的なリスク管理**（ドローダウン制限、トレーリングストップ、部分利確）
- **エントリー条件の厳格化**（複数確認、ダマシフィルター）
- **高度な市場環境フィルター**（市場レジーム検出、エントリー品質評価）
- **ウォークフォワード分析**による過学習防止

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

## テストの実行

ユニットテストを実行して、コードの品質を確認できます：
```bash
python -m pytest tests/ -v
```

## 設定オプション

`.env`ファイルまたはコマンドライン引数で以下のオプションを設定できます：

### 基本設定
- `SYMBOL`: 取引ペア（例: BTCUSDT）
- `SYMBOLS`: 複数ペア（例: BTCUSDT,ETHUSDT,BNBUSDT）
- `INTERVAL`: 時間足（例: 1h, 4h, 1d）
- `START_TIME`/`END_TIME`: バックテスト期間
- `INITIAL_BALANCE`: 初期資金（デフォルト: 10000 USDT）

### 戦略設定
- `SHORT_WINDOW`/`LONG_WINDOW`: 移動平均の期間
- `STOP_LOSS_PERCENT`/`TAKE_PROFIT_PERCENT`: リスク設定
- `MR_RSI_LOWER`/`MR_RSI_UPPER`: RSIの閾値
- `MR_BB_LOWER_FACTOR`/`MR_BB_UPPER_FACTOR`: ボリンジャーバンド乗数
- `STRATEGY_BUY_THRESHOLD`/`STRATEGY_SELL_THRESHOLD`: シグナル閾値

### 戦略の有効/無効
- `ENABLE_MEAN_REVERSION`: Mean Reversion戦略（デフォルト: true）
- `ENABLE_TREND`: トレンド戦略（デフォルト: false）
- `ENABLE_BREAKOUT`: ブレイクアウト戦略（デフォルト: false）

### リスク管理設定
- `MAX_DRAWDOWN_PERCENT`: 最大許容ドローダウン（デフォルト: 10%）
- `DAILY_LOSS_LIMIT_PERCENT`: 日次損失制限（デフォルト: 3%）
- `MAX_CONSECUTIVE_LOSSES`: 連続損失上限（デフォルト: 3回）
- `TRAILING_STOP_ACTIVATION`: トレーリングストップ発動（利益%）
- `TRAILING_STOP_DISTANCE`: トレーリングストップ距離（%）
- `PARTIAL_TP_1`/`PARTIAL_TP_1_RATIO`: 第1部分利確ポイントと比率
- `PARTIAL_TP_2`/`PARTIAL_TP_2_RATIO`: 第2部分利確ポイントと比率

### 市場環境フィルター
- `MIN_ADX_FOR_TREND`: トレンド判定の最小ADX
- `MAX_ADX_FOR_MR`: Mean Reversion用の最大ADX

## プロジェクト構造

```
enhanced_bot/
├── strategies/               # 各戦略のモジュール
│   ├── __init__.py
│   ├── mean_reversion.py     # 強化版平均回帰戦略
│   ├── trend.py              # トレンドフォロー戦略
│   └── breakout.py           # ブレイクアウト戦略
├── tests/                    # ユニットテスト
│   ├── conftest.py           # テストフィクスチャ
│   ├── test_indicators.py    # インジケーターテスト
│   ├── test_strategies.py    # 戦略テスト
│   └── test_config_validation.py  # 設定バリデーションテスト
├── strategy_integrator.py    # 戦略統合モジュール
├── enhanced_trading_bot.py   # メインボットクラス
├── risk_manager.py           # リスク管理モジュール
├── config_validator.py       # 設定バリデーション
├── backtest_runner.py        # バックテスト実行ツール
├── multi_pair_runner.py      # 複数ペア対応ランナー
├── walkforward_analyzer.py   # ウォークフォワード分析
├── visualizer.py             # バックテスト可視化
├── constants.py              # 定数定義
├── parameter_search.py       # グリッドサーチ最適化
├── optuna_search.py          # ベイズ最適化
├── Dockerfile                # Dockerビルド定義
├── docker-compose.yml        # Docker Compose設定
├── requirements.txt          # 依存パッケージ
└── .env                      # 環境変数設定
```

## リスク管理機能

### ドローダウン制限
最大ドローダウンに達すると新規エントリーを停止します。

### 連続損失保護
連続損失後はポジションサイズを自動的に縮小し、上限到達で取引を一時停止します。

### トレーリングストップ
一定の利益に達するとトレーリングストップが発動し、利益を保護します。

### 部分利確
複数段階で利益を確定することで、リスクを低減しながら利益を最大化します。

### 動的SL/TP
ATR（Average True Range）とシグナル強度に基づいて、ストップロスとテイクプロフィットを動的に計算します。

## ウォークフォワード分析

過学習を防ぎ、戦略の実際のパフォーマンスを評価するためのウォークフォワード分析を実行できます：

```bash
python walkforward_analyzer.py
```

分析では以下を行います：
- データを複数のフォールドに分割
- 各フォールドでインサンプル期間を使ってパラメータを最適化
- アウトオブサンプル期間でテスト
- 全期間での一貫性を評価

## 複数ペア対応

複数の取引ペアに対してバックテストを実行し、ポートフォリオ全体のパフォーマンスを分析できます：

```bash
python multi_pair_runner.py
```

## 結果の解析

バックテスト結果は`results`ディレクトリに保存されます：

- バックテスト概要（JSON）
- 取引詳細（CSV）
- 残高履歴（CSV）
- パフォーマンスグラフ（PNG）
- 戦略別パフォーマンス分析（TXT）
- ウォークフォワード分析結果（JSON）

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
