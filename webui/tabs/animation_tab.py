"""
Animation Tab Component

顔アニメーション生成タブの実装
"""

import gradio as gr
from typing import Callable, Optional


def create_animation_tab(generate_callback: Callable):
    """
    アニメーション生成タブを作成
    
    Args:
        generate_callback: アニメーション生成のコールバック関数
    """
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
        
        # イベントハンドラの設定
        generate_btn.click(
            fn=generate_callback,
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
    
        # タブの内容は上記で定義済み
        pass
