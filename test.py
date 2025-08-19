# download_models.py

from audio_separator.separator import Separator
import os

# 출력 디렉토리 설정 (실제로는 중요하지 않음, 모델 캐시를 위한 용도)
output_dir = "/tmp/audio-separator-models"
os.makedirs(output_dir, exist_ok=True)

# Separator 인스턴스 초기화
separator = Separator(output_dir=output_dir)

# 사용할 모델 리스트
models = [
    "Kim_Vocal_1.onnx",
    "UVR_MDXNET_KARA.onnx",
    "UVR-De-Echo-Aggressive.pth",
    "UVR-DeNoise.pth"
]

# 모델을 순차적으로 로드하여 캐시 다운로드 유도
for model_name in models:
    try:
        print(f"📥 Downloading model: {model_name}")
        separator.load_model(model_name)
        print(f"✅ Downloaded: {model_name}")
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
