import os
from audio_separator.separator import Separator

# 입력값 설정
input_audio_path = "sample.mp3"  # 테스트할 로컬 오디오 파일 경로
output_dir = "outputs/test_uvr"     # 출력 디렉토리

os.makedirs(output_dir, exist_ok=True)

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

# 이름 통일
os.rename(os.path.join(output_dir, voc_inst[0]), instrumental_path)
os.rename(os.path.join(output_dir, voc_inst[1]), vocals_path)

# Step 2: Lead / Backing Vocal 분리
print("[Step 2] Lead / Backing Vocal 분리")
separator.load_model("UVR_MDXNET_KARA.onnx")
backing_voc = separator.separate(vocals_path)
os.rename(os.path.join(output_dir, backing_voc[0]), backing_vocals_path)
os.rename(os.path.join(output_dir, backing_voc[1]), lead_vocals_path)

# Step 3: DeReverb (잔향 제거)
print("[Step 3] DeReverb 처리")
separator.load_model("UVR-De-Echo-Aggressive.pth")
voc_no_reverb = separator.separate(lead_vocals_path)
os.rename(os.path.join(output_dir, voc_no_reverb[0]), lead_vocals_no_reverb_path)
os.rename(os.path.join(output_dir, voc_no_reverb[1]), lead_vocals_reverb_path)

# Step 4: Denoise (노이즈 제거)
print("[Step 4] Denoise 처리")
separator.load_model("UVR-DeNoise.pth")
voc_no_noise = separator.separate(lead_vocals_no_reverb_path)
os.rename(os.path.join(output_dir, voc_no_noise[0]), lead_vocals_noise_path)
os.rename(os.path.join(output_dir, voc_no_noise[1]), lead_vocals_no_noise_path)

print("\n✅ 모든 처리가 완료되었습니다. 결과 경로:", output_dir)
