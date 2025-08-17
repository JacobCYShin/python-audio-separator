import os
import json
import base64
import tempfile
import logging
import shutil
import time
from typing import Dict, Any, Optional
import traceback

import runpod
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from audio_separator.separator import Separator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RunPod 업로드 유틸리티 (URL 반환용)
try:
    from runpod.serverless.utils import rp_upload
except Exception:  # 로컬 환경 대비
    rp_upload = None

# AWS S3 설정 (환경변수에서 가져오기)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'likebutter-bucket')

# S3 클라이언트 초기화
def get_s3_client():
    """S3 클라이언트를 반환합니다."""
    try:
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            logger.warning("AWS 인증 정보가 설정되지 않았습니다. 환경변수를 확인해주세요.")
            return None
            
        return boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    except Exception as e:
        logger.error(f"S3 클라이언트 생성 실패: {e}")
        return None

# 전역 변수로 Separator 인스턴스 저장 (Cold start 최적화)
separator = None

def load_separator():
    """Separator 인스턴스를 로드하고 모델을 준비합니다."""
    global separator
    if separator is None:
        try:
            logger.info("Separator 인스턴스를 초기화하고 모델을 로드합니다...")

            # 출력 디렉토리 환경변수 우선 적용
            output_dir = os.getenv("OUTPUT_DIR", "/workspace/output_results/")
            os.makedirs(output_dir, exist_ok=True)

            separator = Separator(
                log_level=logging.INFO,
                model_file_dir="/tmp/audio-separator-models/",
                output_dir=output_dir,
                output_format="MP3",  # WAV에서 MP3로 변경
                normalization_threshold=0.9,
                amplification_threshold=0.0,
                use_autocast=True  # GPU 가속 사용
            )
            logger.info(f"Separator output_dir 설정: {separator.output_dir}")
            
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

def _encode_outputs_as_base64(file_paths: list[str]) -> Dict[str, str]:
    """출력 파일을 base64로 인코딩하여 반환합니다."""
    result_files: Dict[str, str] = {}
    for output_file in file_paths:
        if os.path.exists(output_file):
            try:
                with open(output_file, "rb") as f:
                    file_data = f.read()
                    file_name = os.path.basename(output_file)
                    
                    # 파일 크기 확인 및 압축 고려
                    file_size = len(file_data)
                    logger.info(f"파일 크기: {file_name} = {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
                    
                    # 파일이 너무 크면 경고
                    if file_size > 50 * 1024 * 1024:  # 50MB
                        logger.warning(f"파일이 너무 큽니다: {file_name} ({file_size / 1024 / 1024:.2f} MB)")
                        # 파일을 건너뛰고 경고만 반환
                        result_files[f"{file_name}_SKIPPED"] = "File too large"
                        continue
                    
                    result_files[file_name] = base64.b64encode(file_data).decode('utf-8')
                    logger.info(f"파일 인코딩 완료: {file_name}")
            except Exception as e:
                logger.error(f"파일 인코딩 실패: {output_file} - {e}")
                result_files[f"{os.path.basename(output_file)}_ERROR"] = str(e)
        else:
            logger.warning(f"파일이 존재하지 않습니다: {output_file}")
    return result_files

def _download_from_s3(s3_url: str) -> str:
    """S3 URL에서 파일을 다운로드하여 임시 파일 경로를 반환합니다."""
    try:
        # S3 URL 파싱: https://bucket.s3.region.amazonaws.com/key
        url_parts = s3_url.replace('https://', '').split('/')
        bucket_name = url_parts[0].split('.')[0]
        s3_key = '/'.join(url_parts[1:])
        
        logger.info(f"S3에서 다운로드: bucket={bucket_name}, key={s3_key}")
        
        # S3 클라이언트 가져오기
        s3_client = get_s3_client()
        if s3_client is None:
            raise RuntimeError("S3 클라이언트를 초기화할 수 없습니다")
        
        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file_path = temp_file.name
        temp_file.close()
        
        # S3에서 다운로드
        s3_client.download_file(bucket_name, s3_key, temp_file_path)
        logger.info(f"S3 다운로드 완료: {temp_file_path}")
        
        return temp_file_path
        
    except Exception as e:
        logger.error(f"S3 다운로드 실패: {s3_url} - {e}")
        raise

def _upload_to_s3(file_path: str, file_type: str = "audio") -> str:
    """파일을 S3에 업로드하고 공개 URL을 반환합니다."""
    try:
        # S3 클라이언트 가져오기
        s3_client = get_s3_client()
        if s3_client is None:
            raise RuntimeError("S3 클라이언트를 초기화할 수 없습니다")
        
        # S3 키 생성 (타임스탬프 포함)
        timestamp = int(time.time())
        file_name = os.path.basename(file_path)
        
        if file_type == "audio":
            s3_key = f"generated-audios/{timestamp}_{file_name}"
        else:
            s3_key = f"generated-images/{timestamp}_{file_name}"
        
        # S3에 업로드
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        
        # S3 공개 URL 생성
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        
        logger.info(f"S3 업로드 완료: {file_path} -> {s3_url}")
        return s3_url
        
    except Exception as e:
        logger.error(f"S3 업로드 실패: {file_path} - {e}")
        raise

def _resolve_single_path(path: str, candidate_dirs: list[str]) -> str:
    """상대 경로로 전달된 파일을 후보 디렉토리에서 찾아 절대 경로로 반환."""
    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    # 그대로 존재하는지 먼저 확인
    if os.path.exists(path):
        return os.path.abspath(path)
    for d in candidate_dirs:
        candidate = os.path.join(d, path)
        if os.path.exists(candidate):
            return candidate
    logger.warning(f"파일 경로 해석 실패: {path} (후보: {candidate_dirs})")
    return path

def _resolve_paths(paths: list[str], candidate_dirs: list[str]) -> list[str]:
    return [_resolve_single_path(p, candidate_dirs) for p in paths]

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
    """기본 오디오 분리 처리 (S3 URL 기반)"""
    try:
        # 필수 필드 검증
        if "audio_url" not in job_input:
            return {
                "error": "Missing audio_url",
                "message": "audio_url field is required (S3 URL)"
            }
        
        # 요청 파라미터 추출
        audio_url = job_input["audio_url"]  # S3 URL
        model_filename = job_input.get("model_filename", "Kim_Vocal_1.onnx")
        output_format = job_input.get("output_format", "MP3")
        custom_output_names = job_input.get("custom_output_names", None)
        
        # Separator 인스턴스 로드
        separator_instance = load_separator()
        
        # 요청된 모델 로드
        logger.info(f"모델 로드: {model_filename}")
        separator_instance.load_model(model_filename)
        
        # 출력 형식 설정
        separator_instance.output_format = output_format
        
        # S3에서 오디오 파일 다운로드
        logger.info(f"S3에서 오디오 다운로드: {audio_url}")
        input_file = _download_from_s3(audio_url)
        
        try:
            # 오디오 분리 실행
            logger.info("오디오 분리 시작...")
            output_files = separator_instance.separate(
                input_file, 
                custom_output_names=custom_output_names
            )
            
            logger.info(f"분리 완료. 출력 파일: {output_files}")

            # 반환된 상대 경로를 실제 파일 경로로 해석
            candidate_dirs = [
                getattr(separator_instance, "output_dir", "/tmp/output/"),
                os.getcwd(),
                os.path.dirname(input_file),
            ]
            resolved_outputs = _resolve_paths(output_files, candidate_dirs)
            logger.info(f"해석된 출력 경로: {resolved_outputs}")
            
            # S3에 업로드하고 URL 반환
            output_urls = {}
            for output_file in resolved_outputs:
                if os.path.exists(output_file):
                    s3_url = _upload_to_s3(output_file, "audio")
                    output_urls[os.path.basename(output_file)] = s3_url
            
            return {
                "success": True,
                "message": "Audio separation completed successfully",
                "output_urls": output_urls,
                "model_used": model_filename,
                "input_url": audio_url
            }
            
        finally:
            # 임시 파일 정리
            if os.path.exists(input_file):
                os.unlink(input_file)
                logger.info(f"임시 파일 정리: {input_file}")
            
    except Exception as e:
        logger.error(f"오디오 분리 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": "Audio separation failed",
            "message": str(e)
        }

def handle_advanced_separate(job_input):
    """고급 오디오 분리 처리 (4단계: Vocals/Instrumental, Lead/Backing, DeReverb, Denoise) - S3 URL 기반"""
    try:
        # 필수 필드 검증
        if "audio_url" not in job_input:
            return {
                "error": "Missing audio_url",
                "message": "audio_url field is required (S3 URL)"
            }
        
        # 요청 파라미터 추출
        audio_url = job_input["audio_url"]  # S3 URL
        output_format = job_input.get("output_format", "MP3")
        
        # Separator 인스턴스 로드
        separator_instance = load_separator()
        separator_instance.output_format = output_format
        
        # S3에서 오디오 파일 다운로드
        logger.info(f"S3에서 오디오 다운로드: {audio_url}")
        input_file = _download_from_s3(audio_url)
        
        try:
            logger.info(f"입력 오디오 파일 다운로드 완료: {input_file}")
            
            # Path 해석을 위한 후보 디렉토리
            candidate_dirs = [
                getattr(separator_instance, "output_dir", "/tmp/output/"),
                os.getcwd(),
                os.path.dirname(input_file),
            ]
            
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
            
            # 파일 경로 설정 (이동 없이 생성된 파일 그대로 사용)
            if len(voc_inst) >= 2:
                instrumental_path_raw = voc_inst[0]
                vocals_path_raw = voc_inst[1]
                instrumental_path = _resolve_single_path(instrumental_path_raw, candidate_dirs)
                vocals_path = _resolve_single_path(vocals_path_raw, candidate_dirs)
                logger.info(f"Step 1 파일 경로 설정: {instrumental_path}, {vocals_path}")
            else:
                raise RuntimeError("Step 1 결과 파일이 충분하지 않습니다.")
            
            # Step 2: Lead / Backing Vocal 분리
            logger.info("[Step 2] Lead / Backing Vocal 분리")
            separator_instance.load_model("UVR_MDXNET_KARA.onnx")
            backing_voc = separator_instance.separate(vocals_path)
            logger.info(f"Lead/Backing Vocal 분리 완료: {len(backing_voc)}개 파일 생성")
            
            if len(backing_voc) >= 2:
                backing_vocals_path = _resolve_single_path(backing_voc[0], candidate_dirs)
                lead_vocals_path = _resolve_single_path(backing_voc[1], candidate_dirs)
                logger.info(f"Step 2 파일 경로 설정: {backing_vocals_path}, {lead_vocals_path}")
            else:
                raise RuntimeError("Step 2 결과 파일이 충분하지 않습니다.")
            
            # Step 3: DeReverb (잔향 제거)
            logger.info("[Step 3] DeReverb 처리")
            separator_instance.load_model("UVR-De-Echo-Aggressive.pth")
            voc_no_reverb = separator_instance.separate(lead_vocals_path)
            logger.info(f"DeReverb 처리 완료: {len(voc_no_reverb)}개 파일 생성")
            
            if len(voc_no_reverb) >= 2:
                lead_vocals_no_reverb_path = _resolve_single_path(voc_no_reverb[0], candidate_dirs)
                lead_vocals_reverb_path = _resolve_single_path(voc_no_reverb[1], candidate_dirs)
                logger.info(f"Step 3 파일 경로 설정: {lead_vocals_no_reverb_path}, {lead_vocals_reverb_path}")
            else:
                raise RuntimeError("Step 3 결과 파일이 충분하지 않습니다.")
            
            # Step 4: Denoise (노이즈 제거)
            logger.info("[Step 4] Denoise 처리")
            separator_instance.load_model("UVR-DeNoise.pth")
            voc_no_noise = separator_instance.separate(lead_vocals_no_reverb_path)
            logger.info(f"Denoise 처리 완료: {len(voc_no_noise)}개 파일 생성")
            
            if len(voc_no_noise) >= 2:
                lead_vocals_noise_path = _resolve_single_path(voc_no_noise[0], candidate_dirs)
                lead_vocals_no_noise_path = _resolve_single_path(voc_no_noise[1], candidate_dirs)
                logger.info(f"Step 4 파일 경로 설정: {lead_vocals_noise_path}, {lead_vocals_no_noise_path}")
            else:
                raise RuntimeError("Step 4 결과 파일이 충분하지 않습니다.")
            
            # 결과 반환 방식 분기
            final_output_paths = [
                instrumental_path,
                lead_vocals_no_noise_path
            ]

            # S3에 업로드하고 URL 반환
            output_urls = {}
            for output_file in final_output_paths:
                if os.path.exists(output_file):
                    s3_url = _upload_to_s3(output_file, "audio")
                    output_urls[os.path.basename(output_file)] = s3_url
            
            return {
                "success": True,
                "message": "Advanced audio separation completed successfully",
                "output_urls": output_urls,
                "steps_completed": [
                    "Vocals/Instrumental separation",
                    "Lead/Backing vocal separation", 
                    "DeReverb processing",
                    "Denoise processing"
                ],
                "final_outputs": [
                    "Instrumental.mp3 - 분리된 반주",
                    "Vocals_No_Noise.mp3 - 노이즈 제거된 보컬"
                ],
                "input_url": audio_url
            }
            
        finally:
            # 임시 파일 정리
            if os.path.exists(input_file):
                os.unlink(input_file)
                logger.info(f"임시 파일 정리: {input_file}")
            
    except Exception as e:
        logger.error(f"고급 오디오 분리 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": "Advanced audio separation failed",
            "message": str(e)
        }

# 환경변수 설정 (RunPod 환경에서 사용)
def setup_environment():
    """환경변수를 설정합니다."""
    env_vars = {
        'AWS_REGION': AWS_REGION,
        'S3_BUCKET_NAME': S3_BUCKET_NAME,
        'OUTPUT_DIR': os.getenv('OUTPUT_DIR', '/workspace/output_results/'),
        'PRELOAD_MODELS': os.getenv('PRELOAD_MODELS', 'true'),
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO')
    }
    
    # AWS 인증 정보는 None이 아닐 때만 설정
    if AWS_ACCESS_KEY_ID:
        env_vars['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    if AWS_SECRET_ACCESS_KEY:
        env_vars['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    
    for key, value in env_vars.items():
        if key not in os.environ and value is not None:
            os.environ[key] = value
            logger.info(f"환경변수 설정: {key}")

# 환경변수 설정 실행
setup_environment()

# Cold start 최적화: 컨테이너 시작 시 모델 미리 로드
try:
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        logger.info("컨테이너 시작 시 모델 미리 로드 중...")
        load_separator()
        logger.info("Cold start 최적화 완료")
    else:
        logger.info("PRELOAD_MODELS=false: 런타임 최초 요청 시 로드")
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
