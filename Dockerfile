# ベースイメージ
FROM python:3.10-slim

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y build-essential libatlas-base-dev libffi-dev libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを作成
WORKDIR /app

# 依存関係ファイルをコピー
COPY requirements.txt .

# Pythonパッケージをインストール
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir optuna==3.3.0

# アプリコードをコピー
COPY . .

# ログと結果のディレクトリを作成
RUN mkdir -p logs results cache data

# 実行時の環境変数
ENV PYTHONUNBUFFERED=1

# 起動コマンド（バックテストをデフォルトとする）
ENTRYPOINT ["python", "backtest_runner.py"]

# デフォルトの引数
CMD ["--output", "results"]