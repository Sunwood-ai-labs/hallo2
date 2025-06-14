#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小限のHallo2 Gradio WebUI Application（テスト用）
"""

import os
import argparse
import logging
import gradio as gr

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_function(text_input):
    """シンプルなテスト関数"""
    return f"入力されたテキスト: {text_input}"


def create_simple_interface():
    """最小限のGradioインターフェース"""
    
    # シンプルなインターフェース
    interface = gr.Interface(
        fn=simple_function,
        inputs=gr.Textbox(label="テキストを入力"),
        outputs=gr.Textbox(label="出力"),
        title="Hallo2 テスト",
        description="Gradioの動作確認用"
    )
    
    return interface


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Hallo2 テスト用 WebUI")
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
    
    args = parser.parse_args()
    
    logger.info(f"🎭 Hallo2 テスト用WebUIを起動します...")
    logger.info(f"🌐 サーバー: {args.server_name}:{args.server_port}")
    
    # シンプルなインターフェースを作成
    interface = create_simple_interface()
    
    # 起動
    interface.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_api=False  # APIドキュメント生成を無効化
    )


if __name__ == "__main__":
    main()
