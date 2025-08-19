import os
import numpy as np
from scipy.io.wavfile import write
from audio_separator.separator import Separator

# 입력값 설정
input_audio_path = ""   # 빈 값일 경우 dummy 오디오 생성
output_dir = "/tmp/audio-separator-models"     # 출력 디렉토리
os.makedirs(output_dir, exist_ok=True)

# Dummy 오디오 생성 함수
def generate_dummy_audio(path, duration=3, sr=44100, freq=440):
    """
    duration: 초 단위 길이
    sr: 샘플레이트
    freq: 사인파 주파수 (Hz)
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)  # -1.0~1.0 범위 사인파
    signal_int16 = np.int16(signal * 32767)      # PCM 16비트 변환
    write(path, sr, signal_int16)
    return path

# 입력 경로 확인 → 없으면 dummy 생성
if not input_audio_path or not os.path.exists(input_audio_path):
    print("[!] 입력 오디오가 지정되지 않았습니다. Dummy 오디오 생성 중...")
    input_audio_path = os.path.join(output_dir, "dummy.wav")
    generate_dummy_audio(input_audio_path)

# 출력 파일명 정의
vocals_path = os.path.join(output_dir, 'Vocals.wav')
instrumental_path = os.path.join(output_dir, 'Instrumental.wav')
lead_vocals_path = os.path.join(output_dir, 'Lead_Vocals.wav')
backing_vocals_path = os.path.join(output_dir, 'Backing_Vocals.wav')
lead_vocals_reverb_path = os.path.join(output_dir, 'Vocals_Reverb.wav')
lead_vocals_no_reverb_path = os.path.join(output_dir, 'Vocals_No_Reverb.wav')
lead_vocals_noise_path = os.path.join(output_dir, 'Vocals_Noise.wav')
lead_vocals_no_noise_path = os.path.join(output_dir, 'Vocals_No_Noise.wav')

# Separator 초기화
separator = Separator(output_dir=output_dir)

# Step 1: Vocals / Instrumental 분리
print("[Step 1] Vocals / Instrumental 분리")
try:
    separator.load_model("Kim_Vocal_1.onnx")
except Exception as e:
    print(f"[!] Kim_Vocal_1.onnx 로드 실패 → 대체 모델 사용: {e}")
    separator.load_model("UVR_MDXNET_KARA.onnx")
voc_inst = separator.separate(input_audio_path)

os.rename(os.path.join(output_dir, voc_inst[0]), instrumental_path)
os.rename(os.path.join(output_dir, voc_inst[1]), vocals_path)

# Step 2: Lead / Backing Vocal 분리
print("[Step 2] Lead / Backing Vocal 분리")
separator.load_model("UVR_MDXNET_KARA.onnx")
backing_voc = separator.separate(vocals_path)
os.rename(os.path.join(output_dir, backing_voc[0]), backing_vocals_path)
os.rename(os.path.join(output_dir, backing_voc[1]), lead_vocals_path)

# Step 3: DeReverb
print("[Step 3] DeReverb 처리")
separator.load_model("UVR-De-Echo-Aggressive.pth")
voc_no_reverb = separator.separate(lead_vocals_path)
os.rename(os.path.join(output_dir, voc_no_reverb[0]), lead_vocals_no_reverb_path)
os.rename(os.path.join(output_dir, voc_no_reverb[1]), lead_vocals_reverb_path)

# Step 4: Denoise
print("[Step 4] Denoise 처리")
separator.load_model("UVR-DeNoise.pth")
voc_no_noise = separator.separate(lead_vocals_no_reverb_path)
os.rename(os.path.join(output_dir, voc_no_noise[0]), lead_vocals_noise_path)
os.rename(os.path.join(output_dir, voc_no_noise[1]), lead_vocals_no_noise_path)

print("\n✅ 모든 처리가 완료되었습니다. 결과 경로:", output_dir)
