#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hallo2 Gradio WebUI Application

このアプリケーションは、Hallo2モデルを使用して音声から顔アニメーションを生成するWebインターフェースを提供します。
主な機能:
1. 参照画像のアップロード
2. 音声ファイルのアップロード
3. 顔アニメーション生成
4. ビデオ超解像度処理
"""

import os
import argparse
import logging

# WebUIのインポート
from webui import Hallo2WebUI

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description="Hallo2 Gradio WebUI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference/long.yaml",
        help="設定ファイルのパス"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="サーバーホスト名"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="サーバーポート"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="パブリックリンクを生成"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモード"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 必要なディレクトリの作成
    os.makedirs("output_long/debug", exist_ok=True)
    os.makedirs("hq_results", exist_ok=True)
    os.makedirs(".cache", exist_ok=True)
    
    # WebUIの起動
    webui = Hallo2WebUI(config_path=args.config)
    
    logger.info(f"🎭 Hallo2 WebUIを起動します...")
    logger.info(f"📁 設定ファイル: {args.config}")
    logger.info(f"🌐 サーバー: {args.server_name}:{args.server_port}")
    logger.info(f"📍 アクセスURL: http://{args.server_name}:{args.server_port}")
    
    webui.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
