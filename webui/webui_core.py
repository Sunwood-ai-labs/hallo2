"""
Hallo2 WebUI Core Module

メインのWebUIクラス実装
"""

import os
import sys
import tempfile
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import logging

import gradio as gr
import torch
from omegaconf import OmegaConf

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inference_long import inference_process
from scripts.video_sr import main as video_sr_main
from hallo.utils.util import merge_videos as merge_videos_from_dir

# タブコンポーネントのインポート
from .tabs import create_animation_tab, create_enhancement_tab, create_usage_tab

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
            
            # 一意の一時ファイル名を生成
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            
            # 一時ファイルの作成
            image_path = os.path.join(self.temp_dir, f"source_image_{unique_id}.jpg")
            audio_path = os.path.join(self.temp_dir, f"driving_audio_{unique_id}.wav")
            
            # ファイルのコピー
            if hasattr(source_image, 'name'):
                shutil.copy2(source_image.name, image_path)
            else:
                shutil.copy2(source_image, image_path)
                
            if hasattr(driving_audio, 'name'):
                shutil.copy2(driving_audio.name, audio_path)
            else:
                shutil.copy2(driving_audio, audio_path)
            
            logger.info(f"一時ファイルを作成: 画像={image_path}, 音声={audio_path}")
            
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
            logger.info(f"推論処理を開始: 画像={image_path}, 音声={audio_path}")
            save_seg_path = inference_process(args)
            logger.info(f"推論処理完了: セグメント保存先={save_seg_path}")
            
            progress(0.8, desc="動画を結合中...")
            
            # 生成された動画を結合
            output_video_path = os.path.join(self.temp_dir, "generated_video.mp4")
            
            # セグメント動画ディレクトリが存在し、動画ファイルがある場合
            if os.path.exists(save_seg_path):
                video_files = [f for f in os.listdir(save_seg_path) if f.endswith('.mp4')]
                if video_files:
                    logger.info(f"セグメント動画ディレクトリ: {save_seg_path}")
                    logger.info(f"見つかった動画ファイル: {video_files}")
                    
                    # 動画ファイルが1つの場合は直接コピー
                    if len(video_files) == 1:
                        single_video_path = os.path.join(save_seg_path, video_files[0])
                        shutil.copy2(single_video_path, output_video_path)
                        logger.info(f"単一の動画ファイルをコピー: {single_video_path} -> {output_video_path}")
                    else:
                        # 複数の動画ファイルを結合
                        merge_videos_from_dir(save_seg_path, output_video_path)
                        logger.info(f"複数の動画ファイルを結合: {len(video_files)}個のファイル")
                    
                    progress(1.0, desc="完了!")
                    return output_video_path, "✅ 顔アニメーションの生成が完了しました。"
                else:
                    logger.error(f"セグメント動画ディレクトリに動画ファイルが見つかりません: {save_seg_path}")
                    return None, "❌ 生成された動画ファイルが見つかりません。"
            else:
                logger.error(f"セグメント動画ディレクトリが存在しません: {save_seg_path}")
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
        # Gradioのバージョン問題を回避するため、古い構文を使用
        css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
            padding: 20px !important;
        }
        .tab-nav {
            font-size: 16px;
        }
        .block {
            border-radius: 8px;
        }
        """
        
        # Gradio 4.36.1互換の構文を使用
        interface = gr.Blocks(
            title="Hallo2 - 音声駆動顔アニメーション",
            css=css,
            theme='JohnSmith9982/small_and_pretty'
        )
        
        with interface:
            # ヘッダー
            gr.Markdown("""
            # 🎭 Hallo2 - 音声駆動顔アニメーション
            
            **Hallo2**を使用して、音声から自然な顔アニメーションを生成します。
            参照画像と音声ファイルをアップロードして、リアルなトーキングヘッドビデオを作成できます。
            """)
            
            with gr.Tabs():
                # タブの作成
                create_animation_tab(self.generate_face_animation)
                create_enhancement_tab(self.enhance_video_quality)
                create_usage_tab()
        
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
