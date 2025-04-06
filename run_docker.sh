#!/bin/bash

# Docker を使った拡張版トレーディングボットの実行スクリプト

cd enhanced_bot

# 必要なディレクトリの作成
mkdir -p data logs results cache

# Docker イメージのビルド
echo "Docker イメージをビルドしています..."
docker build -t enhanced-trading-bot .

# コンテナの実行
echo "バックテストを実行しています..."
docker run --rm --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/cache:/app/cache \
  enhanced-trading-bot

echo "バックテスト完了。results ディレクトリで結果を確認してください。"