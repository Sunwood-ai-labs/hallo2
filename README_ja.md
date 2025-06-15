<h1 align='center'>🎬 Hallo2: 長時間・高解像度音声駆動ポートレート画像アニメーション</h1>

<div align='center'>
    <a href='https://github.com/cuijh26' target='_blank'>Jiahao Cui</a><sup>1*</sup>&emsp;
    <a href='https://github.com/crystallee-ai' target='_blank'>Hui Li</a><sup>1*</sup>&emsp;
    <a href='https://yoyo000.github.io/' target='_blank'>Yao Yao</a><sup>3</sup>&emsp;
    <a href='http://zhuhao.cc/home/' target='_blank'>Hao Zhu</a><sup>3</sup>&emsp;
    <a href='https://github.com/NinoNeumann' target='_blank'>Hanlin Shang</a><sup>1</sup>&emsp;
    <a href='https://github.com/Kaihui-Cheng' target='_blank'>Kaihui Cheng</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Hang Zhou</a><sup>2</sup>&emsp;
</div>
<div align='center'>
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>1✉️</sup>&emsp;
    <a href='https://jingdongwang2017.github.io/' target='_blank'>Jingdong Wang</a><sup>2</sup>&emsp;
</div>

<div align='center'>
    <sup>1</sup>復旦大学&emsp; <sup>2</sup>Baidu Inc&emsp; <sup>3</sup>南京大学
</div>

<div align='Center'>
<i><strong><a href='https://iclr.cc/Conferences/2025' target='_blank'>ICLR 2025</a></strong></i>
</div>

<br>
<div align='center'>
    <a href='https://github.com/fudan-generative-vision/hallo2'><img src='https://img.shields.io/github/stars/fudan-generative-vision/hallo2?style=social'></a>
    <a href='https://fudan-generative-vision.github.io/hallo2/#/'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>
    <a href='https://arxiv.org/abs/2410.07718'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/fudan-generative-ai/hallo2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <a href='https://openbayes.com/console/public/tutorials/8KOlYWsdiY4'><img src='https://img.shields.io/badge/Demo-OpenBayes贝式计算-orange'></a>
    <a href='assets/wechat.jpeg'><img src='https://badges.aleen42.com/src/wechat.svg'></a>
</div>

<div align='center'>
    <a href='README.md'>🇺🇸 English</a> | <strong>🇯🇵 日本語</strong>
</div>

<br>

## 📸 ショーケース

<table class="center">
  <tr>
    <td style="text-align: center"><b>テイラー・スウィフトのNYUスピーチ（4K、23分）</b></td>
    <td style="text-align: center"><b>ヨハン・ロックストロームのTEDスピーチ（4K、18分）</b></td>
  </tr>
  <tr>
    <td style="text-align: center"><a target="_blank" href="https://cdn.aondata.work/hallo2/videos/showcases/TailorSpeech.mp4"><img src="https://cdn.aondata.work/hallo2/videos/showcases/gifs/TailorSpeechGIF.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="https://cdn.aondata.work/hallo2/videos/showcases/TEDSpeech.mp4"><img src="https://cdn.aondata.work/hallo2/videos/showcases/gifs/TEDSpeechGIF.gif"></a></td>
  </tr>
  <tr>
    <td style="text-align: center"><b>チャーチルの鉄のカーテンスピーチ（4K、4分）</b></td>
    <td style="text-align: center"><b>スタンフォードのLLMコース（4K、最大1時間）</b></td>
  </tr>
  <tr>
    <td style="text-align: center"><a target="_blank" href="https://cdn.aondata.work/hallo2/videos/showcases/DarkestHour.mp4"><img src="https://cdn.aondata.work/hallo2/videos/showcases/gifs/DarkestHour.gif"></a></td>
    <td style="text-align: center"><a target="_blank" href="https://cdn.aondata.work/hallo2/videos/showcases/LLMCourse.mp4"><img src="https://cdn.aondata.work/hallo2/videos/showcases/gifs/LLMCourseGIF.gif"></a></td>
  </tr>
</table>

詳細なケースについては、[プロジェクトページ](https://fudan-generative-vision.github.io/hallo2/#/)をご覧ください。

## 📰 ニュース

- **`2025/01/23`**: 🎉🎉🎉 我々の論文が[ICLR 2025](https://iclr.cc/Conferences/2025)に採択されました。
- **`2024/10/16`**: ✨✨✨ ソースコードと事前訓練済みの重みがリリースされました。
- **`2024/10/10`**: 🎉🎉🎉 論文が[Arxiv](https://arxiv.org/abs/2410.07718)に投稿されました。

## 📅️ ロードマップ

| ステータス | マイルストーン                                                                                    |    ETA     |
| :--------: | :------------------------------------------------------------------------------------------- | :--------: |
|     ✅     | **[Arixivに論文投稿](https://arxiv.org/abs/2410.07718)**                                      | 2024-10-10 |
|     ✅     | **[GitHubでソースコード公開](https://github.com/fudan-generative-vision/hallo2)**             | 2024-10-16 |
|     🚀     | **[推論性能の加速化]()**                                                                        |    TBD     |

## 🔧️ フレームワーク

![framework](assets/framework_2.jpg)

## ⚙️ インストール

- システム要件: Ubuntu 20.04/Ubuntu 22.04、Cuda 11.8
- テスト済みGPU: A100

コードをダウンロード:

```bash
git clone https://github.com/fudan-generative-vision/hallo2
cd hallo2
```

conda環境を作成:

```bash
conda create -n hallo python=3.10
conda activate hallo
```

`pip`でパッケージをインストール:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

また、ffmpegも必要です:

```bash
apt-get install ffmpeg
```

## 🐳 Dockerデプロイメント

### 📦 GitHub Container Registryから事前ビルド済みイメージを使用

```bash
# プロダクション環境での実行（推奨）
docker run -p 7860:7860 --gpus all \
  -v ./pretrained_models:/app/pretrained_models \
  -v ./examples:/app/examples \
  -v ./output_long:/app/output_long \
  ghcr.io/sunwood-ai-labs/hallo2:latest
```

### 🏗️ ローカルでのDockerビルド

```bash
# Dockerイメージをビルド
docker build -f Dockerfile.cu12 -t hallo2:local .

# 実行
docker run -p 7860:7860 --gpus all hallo2:local
```

### 🚀 Docker Composeを使用（推奨）

```bash
# 開発環境（WebUI起動）
docker-compose up hallo2-webui

# プロダクション環境（事前ビルド済みイメージ使用）
docker-compose --profile production up hallo2-production
```

**アクセス方法:**
- 開発環境: `http://localhost:7865`
- プロダクション環境: `http://localhost:7860`

### 🔄 GitHub Actions CI/CD

このプロジェクトには、GitHub Actionsを使用した自動的なDocker CI/CDパイプラインが含まれています：

- **自動ビルド**: `main`または`master`ブランチへのプッシュ時
- **マルチアーキテクチャサポート**: `linux/amd64`および`linux/arm64`
- **GitHub Container Registry**: 自動的に`ghcr.io`にプッシュ
- **タグ管理**: ブランチ名、セマンティックバージョン、Git SHAによる自動タグ付け

#### 利用可能なイメージタグ:
- `latest`: 最新の安定版
- `main` / `master`: 最新の開発版
- `v1.0.0`: セマンティックバージョンタグ
- `git-abc1234`: 特定のコミットSHA

## 📥 事前訓練済みモデルのダウンロード

推論に必要なすべての事前訓練済みモデルは、我々の[HuggingFace リポジトリ](https://huggingface.co/fudan-generative-ai/hallo2)から簡単に取得できます。

`huggingface-cli`を使用してモデルをダウンロード:

```shell
cd $ProjectRootDir
pip install huggingface_hub
huggingface-cli download fudan-generative-ai/hallo2 --local-dir ./pretrained_models
```

または、各ソースリポジトリから個別にダウンロード:

- [hallo](https://huggingface.co/fudan-generative-ai/hallo2/tree/main/hallo2): 我々のチェックポイント（denoising UNet、face locator、image & audio proj）
- [audio_separator](https://huggingface.co/huangjackson/Kim_Vocal_2): Kim*Vocal_2 MDX-Net ボーカル除去モデル
- [insightface](https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo): 2D・3D顔解析モデル（`pretrained_models/face_analysis/models/`に配置）
- [face landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task): [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models)からの顔検出・メッシュモデル
- [motion module](https://github.com/guoyww/AnimateDiff/blob/main/README.md#202309-animatediff-v2): [AnimateDiff](https://github.com/guoyww/AnimateDiff)からのモーションモジュール
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse): diffusersライブラリ用の重み
- [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5): Stable-Diffusion-v1-2から初期化・ファインチューニング
- [wav2vec](https://huggingface.co/facebook/wav2vec2-base-960h): [Facebook](https://huggingface.co/facebook/wav2vec2-base-960h)のwav音声ベクトル変換モデル
- [facelib](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0): 事前訓練済み顔解析モデル
- [realesrgan](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth): 背景アップサンプルモデル
- [CodeFormer](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0): 事前訓練済み[Codeformer](https://github.com/sczhou/CodeFormer)モデル（ビデオ超解像度モデルをゼロから訓練する場合のみ必要）

最終的に、これらの事前訓練済みモデルは以下のように整理されている必要があります：

```text
./pretrained_models/
|-- audio_separator/
|   |-- download_checks.json
|   |-- mdx_model_data.json
|   |-- vr_model_data.json
|   `-- Kim_Vocal_2.onnx
|-- CodeFormer/
|   |-- codeformer.pth
|   `-- vqgan_code1024.pth
|-- face_analysis/
|   `-- models/
|       |-- face_landmarker_v2_with_blendshapes.task  # mediapipeからの顔ランドマーカーモデル
|       |-- 1k3d68.onnx
|       |-- 2d106det.onnx
|       |-- genderage.onnx
|       |-- glintr100.onnx
|       `-- scrfd_10g_bnkps.onnx
|-- facelib
|   |-- detection_mobilenet0.25_Final.pth
|   |-- detection_Resnet50_Final.pth
|   |-- parsing_parsenet.pth
|   |-- yolov5l-face.pth
|   `-- yolov5n-face.pth
|-- hallo2
|   |-- net_g.pth
|   `-- net.pth
|-- motion_module/
|   `-- mm_sd_v15_v2.ckpt
|-- realesrgan
|   `-- RealESRGAN_x2plus.pth
|-- sd-vae-ft-mse/
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors
|-- stable-diffusion-v1-5/
|   `-- unet/
|       |-- config.json
|       `-- diffusion_pytorch_model.safetensors
`-- wav2vec/
    `-- wav2vec2-base-960h/
        |-- config.json
        |-- feature_extractor_config.json
        |-- model.safetensors
        |-- preprocessor_config.json
        |-- special_tokens_map.json
        |-- tokenizer_config.json
        `-- vocab.json
```

## 🛠️ 推論データの準備

Halloは入力データに対していくつかの簡単な要件があります：

ソース画像について：

1. 正方形にクロップする必要があります
2. 顔が主要な焦点であり、画像の50%-70%を占める必要があります
3. 顔が正面を向いており、回転角度が30°未満である必要があります（横顔は不可）

駆動音声について：

1. WAV形式である必要があります
2. 訓練データセットが英語のみのため、英語である必要があります
3. ボーカルがクリアであることを確認してください（背景音楽は問題ありません）

参考のために[いくつかのサンプル](examples/)を提供しています。

## 🎮 WebUI の実行

Webインターフェースを使用して簡単にHallo2を実行できます：

```bash
python app.py
```

ブラウザで `http://localhost:7860` にアクセスしてWebUIを使用してください。

### オプション引数:

```bash
python app.py --server_name 0.0.0.0 --server_port 7860 --share
```

- `--server_name`: サーバーホスト名（デフォルト: 0.0.0.0）
- `--server_port`: サーバーポート（デフォルト: 7860）
- `--share`: Gradioの共有リンクを生成

## 🎮 推論の実行

### 長時間アニメーション

`scripts/inference_long.py`を実行し、設定ファイル内の`source_image`、`driving_audio`、`save_path`を変更するだけです：

```bash
python scripts/inference_long.py --config ./configs/inference/long.yaml
```

アニメーション結果は`save_path`に保存されます。推論の例については、[examplesフォルダ](https://github.com/fudan-generative-vision/hallo2/tree/main/examples)でより多くの例を見つけることができます。

その他のオプション:

```shell
usage: inference_long.py [-h] [-c CONFIG] [--source_image SOURCE_IMAGE] [--driving_audio DRIVING_AUDIO] [--pose_weight POSE_WEIGHT]
                    [--face_weight FACE_WEIGHT] [--lip_weight LIP_WEIGHT] [--face_expand_ratio FACE_EXPAND_RATIO]

options:
  -h, --help            ヘルプメッセージを表示して終了
  -c CONFIG, --config CONFIG
  --source_image SOURCE_IMAGE
                        ソース画像
  --driving_audio DRIVING_AUDIO
                        駆動音声
  --pose_weight POSE_WEIGHT
                        ポーズの重み
  --face_weight FACE_WEIGHT
                        顔の重み
  --lip_weight LIP_WEIGHT
                        唇の重み
  --face_expand_ratio FACE_EXPAND_RATIO
                        顔領域
```

### 高解像度アニメーション

`scripts/video_sr.py`を実行し、`input_video`と`output_path`を渡すだけです：

```bash
python scripts/video_sr.py --input_path [input_video] --output_path [output_dir] --bg_upsampler realesrgan --face_upsample -w 1 -s 4
```

アニメーション結果は`output_dir`に保存されます。

その他のオプション:

```shell
usage: video_sr.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH] [-w FIDELITY_WEIGHT] [-s UPSCALE] [--has_aligned] [--only_center_face] [--draw_box]
                   [--detection_model DETECTION_MODEL] [--bg_upsampler BG_UPSAMPLER] [--face_upsample] [--bg_tile BG_TILE] [--suffix SUFFIX]

options:
  -h, --help            ヘルプメッセージを表示して終了
  -i INPUT_PATH, --input_path INPUT_PATH
                        入力ビデオ
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        出力フォルダ
  -w FIDELITY_WEIGHT, --fidelity_weight FIDELITY_WEIGHT
                        品質と忠実度のバランス。デフォルト: 0.5
  -s UPSCALE, --upscale UPSCALE
                        画像の最終アップサンプリングスケール。デフォルト: 2
  --has_aligned         入力がクロップ・整列済みの顔。デフォルト: False
  --only_center_face    中央の顔のみを復元。デフォルト: False
  --draw_box            検出された顔の境界ボックスを描画。デフォルト: False
  --detection_model DETECTION_MODEL
                        顔検出器。オプション: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n。デフォルト: retinaface_resnet50
  --bg_upsampler BG_UPSAMPLER
                        背景アップサンプラー。オプション: realesrgan
  --face_upsample       強化後の顔アップサンプラー。デフォルト: False
  --bg_tile BG_TILE     背景サンプラーのタイルサイズ。デフォルト: 400
  --suffix SUFFIX       復元された顔のサフィックス。デフォルト: None
```

> 注意: 高解像度アニメーション機能は[CodeFormer](https://github.com/sczhou/CodeFormer)の修正版です。この機能を使用または再配布する際は、[S-Lab License 1.0](https://github.com/sczhou/CodeFormer?tab=License-1-ov-file)に準拠してください。この使用許諾の条項を尊重していただくことをお願いいたします。

## 🔥 訓練

### 長時間アニメーション

#### 訓練データの準備

推論で使用されるソース画像と同様の話す顔のビデオを利用する訓練データも、以下の要件を満たす必要があります：

1. 正方形にクロップする必要があります
2. 顔が主要な焦点であり、画像の50%-70%を占める必要があります
3. 顔が正面を向いており、回転角度が30°未満である必要があります（横顔は不可）

生のビデオを以下のディレクトリ構造に整理してください：

```text
dataset_name/
|-- videos/
|   |-- 0001.mp4
|   |-- 0002.mp4
|   |-- 0003.mp4
|   `-- 0004.mp4
```

任意の`dataset_name`を使用できますが、`videos`ディレクトリは上記のように命名してください。

次に、以下のコマンドでビデオを処理します：

```bash
python -m scripts.data_preprocess --input_dir dataset_name/videos --step 1
python -m scripts.data_preprocess --input_dir dataset_name/videos --step 2
```

**注意:** ステップ1と2は異なるタスクを実行するため、順次実行してください。ステップ1はビデオをフレームに変換し、各ビデオから音声を抽出し、必要なマスクを生成します。ステップ2はInsightFaceを使用して顔埋め込み、Wav2Vecを使用して音声埋め込みを生成し、GPUが必要です。並列処理には`-p`と`-r`引数を使用してください。

以下のコマンドでメタデータJSONファイルを生成します：

```bash
python scripts/extract_meta_info_stage1.py -r path/to/dataset -n dataset_name
python scripts/extract_meta_info_stage2.py -r path/to/dataset -n dataset_name
```

`path/to/dataset`を上記の例では`dataset_name`など、`videos`の親ディレクトリのパスに置き換えてください。これにより、`./data`ディレクトリに`dataset_name_stage1.json`と`dataset_name_stage2.json`が生成されます。

#### 訓練

設定YAMLファイル`configs/train/stage1.yaml`と`configs/train/stage2_long.yaml`でデータメタパス設定を更新します：

```yaml
#stage1.yaml
data:
  meta_paths:
    - ./data/dataset_name_stage1.json

#stage2.yaml
data:
  meta_paths:
    - ./data/dataset_name_stage2.json
```

以下のコマンドで訓練を開始します：

```shell
accelerate launch -m \
  --config_file accelerate_config.yaml \
  --machine_rank 0 \
  --main_process_ip 0.0.0.0 \
  --main_process_port 20055 \
  --num_machines 1 \
  --num_processes 8 \
  scripts.train_stage1 --config ./configs/train/stage1.yaml
```

##### Accelerate使用説明

`accelerate launch`コマンドは分散設定で訓練プロセスを開始するために使用されます。

```shell
accelerate launch [arguments] {training_script} --{training_script-argument-1} --{training_script-argument-2} ...
```

**Accelerateの引数:**

- `-m, --module`: 起動スクリプトをPythonモジュールとして解釈
- `--config_file`: Hugging Face Accelerateの設定ファイル
- `--machine_rank`: マルチノード設定での現在のマシンのランク
- `--main_process_ip`: マスターノードのIPアドレス
- `--main_process_port`: マスターノードのポート
- `--num_machines`: 訓練に参加するノードの総数
- `--num_processes`: 訓練のための総プロセス数（全マシンの総GPU数と一致）

**訓練の引数:**

- `{training_script}`: 訓練スクリプト（例：`scripts.train_stage1`または`scripts.train_stage2`）
- `--{training_script-argument-1}`: 訓練スクリプト固有の引数。我々の訓練スクリプトは訓練設定ファイルを指定するための`--config`引数を受け取ります

マルチノード訓練の場合、各ノードで異なる`machine_rank`を使用してコマンドを個別に手動実行する必要があります。

詳細設定については、[Accelerateドキュメント](https://huggingface.co/docs/accelerate/en/index)を参照してください。

### 高解像度アニメーション

#### 訓練

##### 訓練データの準備

VFHQデータセットを訓練に使用します。[ホームページ](https://liangbinxie.github.io/projects/vfhq/)からダウンロードできます。その後、`./configs/train/video_sr.yaml`の`dataroot_gt`を更新してください。

#### 訓練

以下のコマンドで訓練を開始します：

```shell
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4322 \
basicsr/train.py -opt ./configs/train/video_sr.yaml \
--launcher pytorch
```

## 📝 引用

この研究が役立つと感じましたら、論文の引用をご検討ください：

```
@misc{cui2024hallo2,
	title={Hallo2: Long-Duration and High-Resolution Audio-driven Portrait Image Animation},
	author={Jiahao Cui and Hui Li and Yao Yao and Hao Zhu and Hanlin Shang and Kaihui Cheng and Hang Zhou and Siyu Zhu and️ Jingdong Wang},
	year={2024},
	eprint={2410.07718},
	archivePrefix={arXiv},
	primaryClass={cs.CV}
}
```

## 🌟 研究機会

**復旦大学 Generative Vision Lab**では複数の研究ポジションを募集しています：

- 研究アシスタント
- ポスドク研究員
- 博士課程候補者
- 修士課程学生

詳細情報については、[siyuzhu@fudan.edu.cn](mailto://siyuzhu@fudan.edu.cn)まで連絡してください。

## ⚠️ 社会的リスクと対策

音声入力によって駆動される肖像画像アニメーション技術の開発は、現実的な肖像画がディープフェイクに悪用される可能性など、倫理的な含意という社会的リスクをもたらします。これらのリスクを軽減するためには、倫理的ガイドラインと責任ある使用慣行を確立することが重要です。個人の画像や音声の使用によりプライバシーと同意の懸念も生じます。これらに対処するには、透明なデータ使用ポリシー、インフォームドコンセント、プライバシー権の保護が必要です。これらのリスクに対処し対策を実施することで、この技術の責任ある倫理的な開発を確保することを目指しています。

## 🤗 謝辞

[magic-animate](https://github.com/magic-research/magic-animate)、[AnimateDiff](https://github.com/guoyww/AnimateDiff)、[ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)、[AniPortrait](https://github.com/Zejun-Yang/AniPortrait)、[Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone)のリポジトリの貢献者に、彼らのオープンな研究と探求に感謝いたします。

オープンソースプロジェクトや関連記事で見落としがありましたら、この特定の作業の謝辞を直ちに補完いたします。

## 👏 コミュニティ貢献者

このプロジェクトをより良くするために貢献してくださったすべての方々に感謝いたします！

<a href="https://github.com/fudan-generative-vision/hallo2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=fudan-generative-vision/hallo2" />
</a>
