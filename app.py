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
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional
import logging

import gradio as gr
import torch
from omegaconf import OmegaConf

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.inference_long import inference_process
from scripts.video_sr import main as video_sr_main
from hallo.utils.util import merge_videos as merge_videos_from_dir

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Hallo2WebUI:
    """
    Hallo2のWebUIを管理するクラス
    """
    
    def __init__(self, config_path: str = "configs/inference/long.yaml"):
        """
        WebUIを初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.temp_dir = tempfile.mkdtemp()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 設定ファイルの読み込み
        try:
            self.config = OmegaConf.load(config_path)
            logger.info(f"設定ファイルを読み込みました: {config_path}")
        except Exception as e:
            logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
            raise
            
    def generate_face_animation(
        self,
        source_image,
        driving_audio,
        pose_weight: float = 1.0,
        face_weight: float = 1.0,
        lip_weight: float = 1.0,
        face_expand_ratio: float = 1.2,
        progress=gr.Progress()
    ) -> Tuple[Optional[str], str]:
        """
        顔アニメーションを生成
        
        Args:
            source_image: 参照画像
            driving_audio: 音声ファイル
            pose_weight: ポーズの重み
            face_weight: 顔の重み
            lip_weight: 唇の重み
            face_expand_ratio: 顔の拡張比率
            progress: 進捗表示
            
        Returns:
            生成された動画のパスとステータスメッセージ
        """
        try:
            if source_image is None:
                return None, "❌ 参照画像をアップロードしてください。"
                
            if driving_audio is None:
                return None, "❌ 音声ファイルをアップロードしてください。"
            
            progress(0.1, desc="初期化中...")
            
            # 一時ファイルの作成
            image_path = os.path.join(self.temp_dir, "source_image.jpg")
            audio_path = os.path.join(self.temp_dir, "driving_audio.wav")
            
            # ファイルのコピー
            if hasattr(source_image, 'name'):
                shutil.copy2(source_image.name, image_path)
            else:
                shutil.copy2(source_image, image_path)
                
            if hasattr(driving_audio, 'name'):
                shutil.copy2(driving_audio.name, audio_path)
            else:
                shutil.copy2(driving_audio, audio_path)
            
            progress(0.2, desc="設定を準備中...")
            
            # 引数の準備
            args = argparse.Namespace(
                config=self.config_path,
                source_image=image_path,
                driving_audio=audio_path,
                pose_weight=pose_weight,
                face_weight=face_weight,
                lip_weight=lip_weight,
                face_expand_ratio=face_expand_ratio
            )
            
            progress(0.3, desc="推論処理を開始...")
            
            # 推論実行
            save_seg_path = inference_process(args)
            
            progress(0.8, desc="動画を結合中...")
            
            # 生成された動画を結合
            output_video_path = os.path.join(self.temp_dir, "generated_video.mp4")
            video_segments = []
            
            # セグメント動画ファイルを取得
            if os.path.exists(save_seg_path):
                for file in sorted(os.listdir(save_seg_path)):
                    if file.endswith('.mp4'):
                        video_segments.append(os.path.join(save_seg_path, file))
            
            if video_segments:
                # 動画を結合
                merge_videos_from_dir(video_segments, output_video_path, audio_path)
                progress(1.0, desc="完了!")
                return output_video_path, "✅ 顔アニメーションの生成が完了しました。"
            else:
                return None, "❌ 動画の生成に失敗しました。"
                
        except Exception as e:
            logger.error(f"顔アニメーション生成エラー: {e}")
            return None, f"❌ エラーが発生しました: {str(e)}"
    
    def enhance_video_quality(
        self,
        input_video,
        fidelity_weight: float = 0.5,
        upscale_factor: int = 2,
        face_upsample: bool = True,
        bg_upsampler: str = "realesrgan",
        progress=gr.Progress()
    ) -> Tuple[Optional[str], str]:
        """
        動画の品質を向上
        
        Args:
            input_video: 入力動画
            fidelity_weight: 忠実度の重み
            upscale_factor: アップスケール倍率
            face_upsample: 顔のアップサンプリング
            bg_upsampler: 背景アップサンプラー
            progress: 進捗表示
            
        Returns:
            品質向上後の動画のパスとステータスメッセージ
        """
        try:
            if input_video is None:
                return None, "❌ 動画ファイルを選択してください。"
            
            progress(0.1, desc="動画品質向上処理を開始...")
            
            # 出力パスの設定
            output_path = os.path.join(self.temp_dir, "enhanced_video.mp4")
            
            # video_sr.pyの引数を準備
            sr_args = argparse.Namespace(
                input_path=input_video if isinstance(input_video, str) else input_video.name,
                output_path=output_path,
                fidelity_weight=fidelity_weight,
                upscale=upscale_factor,
                has_aligned=False,
                only_center_face=False,
                draw_box=False,
                detection_model='retinaface_resnet50',
                bg_upsampler=bg_upsampler,
                face_upsample=face_upsample,
                bg_tile=400,
                suffix=None
            )
            
            progress(0.3, desc="品質向上処理を実行中...")
            
            # ビデオ超解像度処理を実行
            # 注意: video_sr.pyは直接呼び出しが困難なため、コマンドライン実行を使用
            import subprocess
            cmd = [
                sys.executable, "scripts/video_sr.py",
                "--input_path", sr_args.input_path,
                "--output_path", sr_args.output_path,
                "--fidelity_weight", str(sr_args.fidelity_weight),
                "--upscale", str(sr_args.upscale),
                "--bg_upsampler", sr_args.bg_upsampler
            ]
            
            if sr_args.face_upsample:
                cmd.append("--face_upsample")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                progress(1.0, desc="完了!")
                return output_path, "✅ 動画品質の向上が完了しました。"
            else:
                logger.error(f"Video SR error: {result.stderr}")
                return None, f"❌ 品質向上処理でエラーが発生しました: {result.stderr}"
                
        except Exception as e:
            logger.error(f"動画品質向上エラー: {e}")
            return None, f"❌ エラーが発生しました: {str(e)}"
    
    def create_interface(self):
        """
        Gradioインターフェースを作成
        """
        with gr.Blocks(
            title="Hallo2 - 音声駆動顔アニメーション",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .tab-nav {
                font-size: 16px;
            }
            """
        ) as interface:
            
            # ヘッダー
            gr.Markdown("""
            # 🎭 Hallo2 - 音声駆動顔アニメーション
            
            **Hallo2**を使用して、音声から自然な顔アニメーションを生成します。
            参照画像と音声ファイルをアップロードして、リアルなトーキングヘッドビデオを作成できます。
            """)
            
            with gr.Tabs():
                # メイン生成タブ
                with gr.Tab("🎬 アニメーション生成", elem_id="main-tab"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📸 参照画像")
                            source_image = gr.Image(
                                label="参照画像をアップロード",
                                type="filepath",
                                height=300
                            )
                            
                            gr.Markdown("### 🎵 音声ファイル")
                            driving_audio = gr.Audio(
                                label="音声ファイルをアップロード",
                                type="filepath"
                            )
                            
                            with gr.Accordion("⚙️ 詳細設定", open=False):
                                pose_weight = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="ポーズの重み"
                                )
                                face_weight = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="顔の重み"
                                )
                                lip_weight = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="唇の重み"
                                )
                                face_expand_ratio = gr.Slider(
                                    minimum=1.0,
                                    maximum=2.0,
                                    value=1.2,
                                    step=0.1,
                                    label="顔の拡張比率"
                                )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎥 生成結果")
                            output_video = gr.Video(
                                label="生成された動画",
                                height=400
                            )
                            
                            status_text = gr.Textbox(
                                label="ステータス",
                                interactive=False,
                                value="準備完了"
                            )
                            
                            generate_btn = gr.Button(
                                "🚀 アニメーション生成",
                                variant="primary",
                                size="lg"
                            )
                
                # 品質向上タブ
                with gr.Tab("✨ 品質向上", elem_id="enhance-tab"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📹 動画アップロード")
                            input_video = gr.Video(
                                label="品質向上したい動画",
                                height=300
                            )
                            
                            with gr.Accordion("🔧 品質向上設定", open=True):
                                fidelity_weight = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.1,
                                    label="忠実度の重み"
                                )
                                upscale_factor = gr.Slider(
                                    minimum=1,
                                    maximum=4,
                                    value=2,
                                    step=1,
                                    label="アップスケール倍率"
                                )
                                face_upsample = gr.Checkbox(
                                    value=True,
                                    label="顔のアップサンプリング"
                                )
                                bg_upsampler = gr.Dropdown(
                                    choices=["None", "realesrgan"],
                                    value="realesrgan",
                                    label="背景アップサンプラー"
                                )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 🎬 品質向上結果")
                            enhanced_video = gr.Video(
                                label="品質向上後の動画",
                                height=400
                            )
                            
                            enhance_status = gr.Textbox(
                                label="ステータス",
                                interactive=False,
                                value="準備完了"
                            )
                            
                            enhance_btn = gr.Button(
                                "✨ 品質向上",
                                variant="primary",
                                size="lg"
                            )
                
                # 使用方法タブ
                with gr.Tab("📖 使用方法"):
                    gr.Markdown("""
                    ## 🚀 使用方法
                    
                    ### 1. アニメーション生成
                    1. **参照画像**をアップロード（人物の顔が写った画像）
                    2. **音声ファイル**をアップロード（WAVまたはMP3形式）
                    3. 必要に応じて**詳細設定**を調整
                    4. **アニメーション生成**ボタンをクリック
                    
                    ### 2. 品質向上
                    1. 生成された動画または任意の動画をアップロード
                    2. **品質向上設定**を調整
                    3. **品質向上**ボタンをクリック
                    
                    ## ⚙️ パラメータ説明
                    
                    ### アニメーション生成パラメータ
                    - **ポーズの重み**: 頭部の動きの強さ（0.0-2.0）
                    - **顔の重み**: 顔の表情の強さ（0.0-2.0）
                    - **唇の重み**: 口の動きの強さ（0.0-2.0）
                    - **顔の拡張比率**: 顔領域の拡張比率（1.0-2.0）
                    
                    ### 品質向上パラメータ
                    - **忠実度の重み**: 品質と忠実度のバランス（0.0-1.0）
                    - **アップスケール倍率**: 解像度の向上倍率（1-4倍）
                    - **顔のアップサンプリング**: 顔領域の品質向上
                    - **背景アップサンプラー**: 背景の品質向上手法
                    
                    ## 💡 ヒント
                    - 高品質な参照画像を使用することで、より良い結果が得られます
                    - 音声は16kHzのサンプリングレートが推奨されます
                    - 処理時間は音声の長さに比例します
                    - GPUを使用することで高速化できます
                    """)
            
            # イベントハンドラの設定
            generate_btn.click(
                fn=self.generate_face_animation,
                inputs=[
                    source_image,
                    driving_audio,
                    pose_weight,
                    face_weight,
                    lip_weight,
                    face_expand_ratio
                ],
                outputs=[output_video, status_text],
                show_progress=True
            )
            
            enhance_btn.click(
                fn=self.enhance_video_quality,
                inputs=[
                    input_video,
                    fidelity_weight,
                    upscale_factor,
                    face_upsample,
                    bg_upsampler
                ],
                outputs=[enhanced_video, enhance_status],
                show_progress=True
            )
            
            # サンプルの設定
            gr.Examples(
                examples=[
                    ["examples/reference_images/1.jpg", "examples/driving_audios/1.wav"],
                    ["examples/reference_images/2.jpg", "examples/driving_audios/2.wav"],
                    ["examples/reference_images/3.jpg", "examples/driving_audios/3.wav"],
                ],
                inputs=[source_image, driving_audio],
                label="📁 サンプル"
            )
        
        return interface
    
    def launch(self, **kwargs):
        """
        WebUIを起動
        """
        interface = self.create_interface()
        interface.launch(**kwargs)
    
    def __del__(self):
        """
        一時ディレクトリのクリーンアップ
        """
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


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
