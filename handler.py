import os
import json
import base64
import tempfile
import logging
import shutil
from typing import Dict, Any, Optional
import traceback

import runpod
from audio_separator.separator import Separator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수로 Separator 인스턴스 저장 (Cold start 최적화)
separator = None

def load_separator():
    """Separator 인스턴스를 로드하고 모델을 준비합니다."""
    global separator
    if separator is None:
        try:
            logger.info("Separator 인스턴스를 초기화하고 모델을 로드합니다...")
            separator = Separator(
                log_level=logging.INFO,
                model_file_dir="/tmp/audio-separator-models/",
                output_dir="/tmp/output/",
                output_format="WAV",
                normalization_threshold=0.9,
                amplification_threshold=0.0,
                use_autocast=True  # GPU 가속 사용
            )
            
            # 필요한 모델들이 다운로드되어 있는지 확인하고 없으면 다운로드
            required_models = [
                'Kim_Vocal_1.onnx',  # Step 1: Vocals/Instrumental 분리
                'UVR_MDXNET_KARA.onnx',  # Step 2: Lead/Backing 분리
                'UVR-De-Echo-Aggressive.pth',  # Step 3: DeReverb
                'UVR-DeNoise.pth'  # Step 4: Denoise
            ]
            
            for model in required_models:
                model_path = os.path.join("/tmp/audio-separator-models/", model)
                if not os.path.exists(model_path):
                    logger.info(f"모델이 없습니다. 다운로드 중: {model}")
                    try:
                        separator.download_model_and_data(model)
                        logger.info(f"모델 다운로드 완료: {model}")
                    except Exception as e:
                        logger.warning(f"모델 다운로드 실패: {model} - {e}")
            
            # 기본 모델 로드 (Kim_Vocal_1.onnx 사용)
            separator.load_model("Kim_Vocal_1.onnx")
            logger.info("모델 로딩 완료")
        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            raise
    return separator

def handler(job):
    """
    RunPod Serverless 핸들러 함수
    
    Args:
        job: RunPod에서 전달하는 작업 데이터
        
    Returns:
        Dict: 처리 결과
    """
    try:
        job_input = job.get("input", {})
        logger.info(f"작업 입력: {job_input}")
        
        # 작업 타입 확인
        job_type = job_input.get("type", "separate")
        
        if job_type == "list_models":
            return handle_list_models()
        elif job_type == "separate":
            return handle_separate_audio(job_input)
        elif job_type == "advanced_separate":
            return handle_advanced_separate(job_input)
        else:
            return {
                "error": f"Unknown job type: {job_type}",
                "message": "Supported types: 'list_models', 'separate', 'advanced_separate'"
            }
            
    except Exception as e:
        logger.error(f"핸들러 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": "Internal server error",
            "message": str(e)
        }

def handle_list_models():
    """모델 목록 조회 처리"""
    try:
        separator_instance = load_separator()
        
        # 모델 목록 가져오기
        models = separator_instance.get_simplified_model_list()
        
        return {
            "success": True,
            "models": models,
            "message": "Available models retrieved successfully"
        }
    except Exception as e:
        logger.error(f"모델 목록 조회 오류: {str(e)}")
        return {
            "error": "Failed to retrieve models",
            "message": str(e)
        }

def handle_separate_audio(job_input):
    """기본 오디오 분리 처리"""
    try:
        # 필수 필드 검증
        if "audio_data" not in job_input:
            return {
                "error": "Missing audio_data",
                "message": "audio_data field is required"
            }
        
        # 요청 파라미터 추출
        audio_data = job_input["audio_data"]  # base64 인코딩된 오디오 데이터
        model_filename = job_input.get("model_filename", "Kim_Vocal_1.onnx")  # 기본 모델 변경
        output_format = job_input.get("output_format", "WAV")
        custom_output_names = job_input.get("custom_output_names", None)
        
        # Separator 인스턴스 로드
        separator_instance = load_separator()
        
        # 모델이 다르면 새로 로드
        if separator_instance.model_filename != model_filename:
            logger.info(f"새 모델 로드: {model_filename}")
            separator_instance.load_model(model_filename)
        
        # 출력 형식 설정
        separator_instance.output_format = output_format
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            # base64 디코딩하여 임시 파일 생성
            audio_bytes = base64.b64decode(audio_data)
            input_file = os.path.join(temp_dir, "input.wav")
            
            with open(input_file, "wb") as f:
                f.write(audio_bytes)
            
            logger.info(f"오디오 파일 생성: {input_file}")
            
            # 오디오 분리 실행
            logger.info("오디오 분리 시작...")
            output_files = separator_instance.separate(
                input_file, 
                custom_output_names=custom_output_names
            )
            
            logger.info(f"분리 완료. 출력 파일: {output_files}")
            
            # 결과 파일들을 base64로 인코딩
            result_files = {}
            for output_file in output_files:
                if os.path.exists(output_file):
                    with open(output_file, "rb") as f:
                        file_data = f.read()
                        file_name = os.path.basename(output_file)
                        result_files[file_name] = base64.b64encode(file_data).decode('utf-8')
            
            return {
                "success": True,
                "message": "Audio separation completed successfully",
                "output_files": result_files,
                "model_used": model_filename
            }
            
    except Exception as e:
        logger.error(f"오디오 분리 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": "Audio separation failed",
            "message": str(e)
        }

def handle_advanced_separate(job_input):
    """고급 오디오 분리 처리 (4단계: Vocals/Instrumental, Lead/Backing, DeReverb, Denoise)"""
    try:
        # 필수 필드 검증
        if "audio_data" not in job_input:
            return {
                "error": "Missing audio_data",
                "message": "audio_data field is required"
            }
        
        # 요청 파라미터 추출
        audio_data = job_input["audio_data"]
        output_format = job_input.get("output_format", "WAV")
        
        # Separator 인스턴스 로드
        separator_instance = load_separator()
        separator_instance.output_format = output_format
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            # base64 디코딩하여 임시 파일 생성
            audio_bytes = base64.b64decode(audio_data)
            input_file = os.path.join(temp_dir, "input.wav")
            
            with open(input_file, "wb") as f:
                f.write(audio_bytes)
            
            logger.info(f"입력 오디오 파일 생성: {input_file}")
            
            # 출력 파일 경로 정의
            vocals_path = os.path.join(temp_dir, "Vocals.wav")
            instrumental_path = os.path.join(temp_dir, "Instrumental.wav")
            lead_vocals_path = os.path.join(temp_dir, "Lead_Vocals.wav")
            backing_vocals_path = os.path.join(temp_dir, "Backing_Vocals.wav")
            lead_vocals_reverb_path = os.path.join(temp_dir, "Vocals_Reverb.wav")
            lead_vocals_no_reverb_path = os.path.join(temp_dir, "Vocals_No_Reverb.wav")
            lead_vocals_noise_path = os.path.join(temp_dir, "Vocals_Noise.wav")
            lead_vocals_no_noise_path = os.path.join(temp_dir, "Vocals_No_Noise.wav")
            
            # Step 1: Vocals / Instrumental 분리
            logger.info("[Step 1] Vocals / Instrumental 분리")
            try:
                separator_instance.load_model("Kim_Vocal_1.onnx")
                logger.info("Kim_Vocal_1.onnx 모델 로드 성공")
            except Exception as e:
                logger.warning(f"Kim_Vocal_1.onnx 로드 실패 → 대체 모델 사용: {e}")
                separator_instance.load_model("UVR_MDXNET_KARA.onnx")
                logger.info("UVR_MDXNET_KARA.onnx 모델 로드 성공")
            
            voc_inst = separator_instance.separate(input_file)
            logger.info(f"Vocals/Instrumental 분리 완료: {len(voc_inst)}개 파일 생성")
            
            # 이름 통일 (shutil.move 사용)
            if len(voc_inst) >= 2:
                shutil.move(voc_inst[0], instrumental_path)
                shutil.move(voc_inst[1], vocals_path)
                logger.info("Step 1 파일명 변경 완료")
            
            # Step 2: Lead / Backing Vocal 분리
            logger.info("[Step 2] Lead / Backing Vocal 분리")
            separator_instance.load_model("UVR_MDXNET_KARA.onnx")
            backing_voc = separator_instance.separate(vocals_path)
            logger.info(f"Lead/Backing Vocal 분리 완료: {len(backing_voc)}개 파일 생성")
            
            if len(backing_voc) >= 2:
                shutil.move(backing_voc[0], backing_vocals_path)
                shutil.move(backing_voc[1], lead_vocals_path)
                logger.info("Step 2 파일명 변경 완료")
            
            # Step 3: DeReverb (잔향 제거)
            logger.info("[Step 3] DeReverb 처리")
            separator_instance.load_model("UVR-De-Echo-Aggressive.pth")
            voc_no_reverb = separator_instance.separate(lead_vocals_path)
            logger.info(f"DeReverb 처리 완료: {len(voc_no_reverb)}개 파일 생성")
            
            if len(voc_no_reverb) >= 2:
                shutil.move(voc_no_reverb[0], lead_vocals_no_reverb_path)
                shutil.move(voc_no_reverb[1], lead_vocals_reverb_path)
                logger.info("Step 3 파일명 변경 완료")
            
            # Step 4: Denoise (노이즈 제거)
            logger.info("[Step 4] Denoise 처리")
            separator_instance.load_model("UVR-DeNoise.pth")
            voc_no_noise = separator_instance.separate(lead_vocals_no_reverb_path)
            logger.info(f"Denoise 처리 완료: {len(voc_no_noise)}개 파일 생성")
            
            if len(voc_no_noise) >= 2:
                shutil.move(voc_no_noise[0], lead_vocals_noise_path)
                shutil.move(voc_no_noise[1], lead_vocals_no_noise_path)
                logger.info("Step 4 파일명 변경 완료")
            
            # 결과 파일들을 base64로 인코딩 (최종 결과만 반환)
            result_files = {}
            output_files = [
                ("Instrumental.wav", instrumental_path),
                ("Vocals_No_Noise.wav", lead_vocals_no_noise_path)
            ]
            
            for filename, file_path in output_files:
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                        result_files[filename] = base64.b64encode(file_data).decode('utf-8')
            
            return {
                "success": True,
                "message": "Advanced audio separation completed successfully",
                "output_files": result_files,
                "steps_completed": [
                    "Vocals/Instrumental separation",
                    "Lead/Backing vocal separation", 
                    "DeReverb processing",
                    "Denoise processing"
                ],
                "final_outputs": [
                    "Instrumental.wav - 분리된 반주",
                    "Vocals_No_Noise.wav - 노이즈 제거된 보컬"
                ]
            }
            
    except Exception as e:
        logger.error(f"고급 오디오 분리 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": "Advanced audio separation failed",
            "message": str(e)
        }

# Cold start 최적화: 컨테이너 시작 시 모델 미리 로드
try:
    logger.info("컨테이너 시작 시 모델 미리 로드 중...")
    load_separator()
    logger.info("Cold start 최적화 완료")
except Exception as e:
    logger.error(f"Cold start 최적화 실패: {str(e)}")

# 로컬 테스트용 함수
def test_local():
    """로컬 테스트용 함수"""
    print("=== 로컬 테스트 시작 ===")
    
    # 모델 목록 조회 테스트
    print("1. 모델 목록 조회 테스트")
    try:
        result = handle_list_models()
        print(f"결과: {result}")
    except Exception as e:
        print(f"오류: {e}")
    
    print("\n=== 로컬 테스트 완료 ===")

# RunPod Serverless 시작
if __name__ == "__main__":
    # 로컬 테스트 모드 확인
    if os.getenv("LOCAL_TEST", "false").lower() == "true":
        test_local()
    else:
        runpod.serverless.start({"handler": handler})
