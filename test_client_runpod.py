#!/usr/bin/env python3
"""
Audio Separator RunPod Serverless API 테스트 클라이언트

이 스크립트는 RunPod Serverless에 배포된 Audio Separator API를 테스트합니다.
"""

import requests
import base64
import json
import os
import sys
from typing import Dict, Any, Optional

class AudioSeparatorRunPodClient:
    """Audio Separator RunPod API 클라이언트"""
    
    def __init__(self, api_url: str, api_key: str = None):
        """
        클라이언트 초기화
        
        Args:
            api_url: RunPod API URL (예: https://your-endpoint.runpod.net)
            api_key: RunPod API 키 (선택사항)
        """
        # API URL 정리 (끝의 슬래시 제거)
        api_url = api_url.rstrip('/')
        
        # /run이 없으면 추가
        if not api_url.endswith('/run'):
            self.api_url = api_url + '/run'
        else:
            self.api_url = api_url
            
        self.api_key = api_key
        self.session = requests.Session()
        
        # 연결 타임아웃 설정 (30초)
        self.session.timeout = 30
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def list_models(self) -> Dict[str, Any]:
        """
        사용 가능한 모델 목록을 조회합니다.
        
        Returns:
            API 응답 데이터
        """
        try:
            payload = {
                "input": {
                    "type": "list_models"
                }
            }
            
            response = self.session.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"모델 목록 조회 실패: {e}")
            return {"error": str(e)}
    
    def separate_audio(
        self,
        audio_file_path: str,
        model_filename: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        output_format: str = "WAV",
        custom_output_names: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        기본 오디오 파일을 분리합니다.
        
        Args:
            audio_file_path: 입력 오디오 파일 경로
            model_filename: 사용할 모델 파일명
            output_format: 출력 형식
            custom_output_names: 출력 파일명 커스터마이징
            
        Returns:
            API 응답 데이터
        """
        try:
            # 오디오 파일을 base64로 인코딩
            with open(audio_file_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            
            # 요청 데이터 구성
            payload = {
                "input": {
                    "type": "separate",
                    "audio_data": audio_data,
                    "model_filename": model_filename,
                    "output_format": output_format
                }
            }
            
            if custom_output_names:
                payload["input"]["custom_output_names"] = custom_output_names
            
            # API 호출
            response = self.session.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()
            
        except FileNotFoundError:
            print(f"오디오 파일을 찾을 수 없습니다: {audio_file_path}")
            return {"error": f"File not found: {audio_file_path}"}
        except requests.exceptions.RequestException as e:
            print(f"오디오 분리 실패: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            return {"error": str(e)}
    
    def advanced_separate_audio(
        self,
        audio_file_path: str,
        output_format: str = "WAV"
    ) -> Dict[str, Any]:
        """
        고급 오디오 분리 (3단계: Vocals/Instrumental, Lead/Backing, DeReverb, Denoise)
        
        Args:
            audio_file_path: 입력 오디오 파일 경로
            output_format: 출력 형식
            
        Returns:
            API 응답 데이터
        """
        try:
            # 오디오 파일을 base64로 인코딩
            with open(audio_file_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            
            print(f"오디오 파일 크기: {len(audio_data)} characters (base64)")
            
            # 요청 데이터 구성
            payload = {
                "input": {
                    "type": "advanced_separate",
                    "audio_data": audio_data,
                    "output_format": output_format
                }
            }
            
            print(f"API URL: {self.api_url}")
            print(f"요청 데이터 크기: {len(str(payload))} characters")
            
            # API 호출
            response = self.session.post(f"{self.api_url}", json=payload)
            print(f"응답 상태 코드: {response.status_code}")
            
            response.raise_for_status()
            return response.json()
            
        except FileNotFoundError:
            print(f"오디오 파일을 찾을 수 없습니다: {audio_file_path}")
            return {"error": f"File not found: {audio_file_path}"}
        except requests.exceptions.Timeout:
            print(f"요청 타임아웃: {self.api_url}")
            return {"error": "Request timeout"}
        except requests.exceptions.ConnectionError as e:
            print(f"연결 오류: {e}")
            return {"error": f"Connection error: {e}"}
        except requests.exceptions.RequestException as e:
            print(f"고급 오디오 분리 실패: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            return {"error": str(e)}
    
    def save_output_files(self, response_data: Dict[str, Any], output_dir: str = ".") -> bool:
        """
        API 응답에서 출력 파일들을 저장합니다.
        
        Args:
            response_data: API 응답 데이터
            output_dir: 출력 디렉토리
            
        Returns:
            성공 여부
        """
        try:
            if "output_files" not in response_data:
                print("출력 파일이 응답에 없습니다.")
                return False
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 각 출력 파일 저장
            for filename, file_data in response_data["output_files"].items():
                output_path = os.path.join(output_dir, filename)
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(file_data))
                print(f"파일 저장됨: {output_path}")
            
            # 고급 분리인 경우 최종 출력 정보 표시
            if "final_outputs" in response_data:
                print("\n최종 출력 파일:")
                for output_info in response_data["final_outputs"]:
                    print(f"  - {output_info}")
            
            return True
            
        except Exception as e:
            print(f"파일 저장 실패: {e}")
            return False

def main():
    """메인 함수"""
    if len(sys.argv) < 3:
        print("사용법: python test_client_runpod.py <API_URL> <AUDIO_FILE> [API_KEY] [--advanced]")
        print("예시: python test_client_runpod.py https://your-endpoint.runpod.net input.wav")
        print("예시: python test_client_runpod.py https://your-endpoint.runpod.net input.wav your-api-key")
        print("예시: python test_client_runpod.py https://your-endpoint.runpod.net input.wav your-api-key --advanced")
        sys.exit(1)
    
    api_url = sys.argv[1]
    audio_file = sys.argv[2]
    api_key = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
    use_advanced = "--advanced" in sys.argv
    
    # 클라이언트 초기화
    client = AudioSeparatorRunPodClient(api_url, api_key)
    
    print("=== Audio Separator RunPod API 테스트 ===")
    print(f"API URL: {api_url}")
    print(f"오디오 파일: {audio_file}")
    print(f"고급 분리 모드: {use_advanced}")
    print()
    
    # 1. 모델 목록 조회
    print("1. 모델 목록 조회 중...")
    models_response = client.list_models()
    
    if "error" in models_response:
        print(f"모델 목록 조회 실패: {models_response['error']}")
        sys.exit(1)
    
    print("사용 가능한 모델:")
    for model in models_response.get("models", [])[:5]:  # 처음 5개만 표시
        print(f"  - {model['model_filename']} ({model['friendly_name']})")
    print()
    
    # 2. 오디오 분리
    print("2. 오디오 분리 중...")
    if use_advanced:
        print("고급 분리 모드 사용 (3단계 처리)")
        separation_response = client.advanced_separate_audio(audio_file)
    else:
        print("기본 분리 모드 사용")
        separation_response = client.separate_audio(
            audio_file,
            custom_output_names={
                "Vocals": "vocals_output",
                "Instrumental": "instrumental_output"
            }
        )
    
    if "error" in separation_response:
        print(f"오디오 분리 실패: {separation_response['error']}")
        sys.exit(1)
    
    print("오디오 분리 완료!")
    if use_advanced:
        print(f"완료된 단계: {separation_response.get('steps_completed', [])}")
        print(f"최종 출력: {separation_response.get('final_outputs', [])}")
    else:
        print(f"사용된 모델: {separation_response.get('model_used', 'Unknown')}")
    print(f"출력 파일 수: {len(separation_response.get('output_files', {}))}")
    print()
    
    # 3. 결과 파일 저장
    print("3. 결과 파일 저장 중...")
    output_dir = "output_advanced" if use_advanced else "output"
    if client.save_output_files(separation_response, output_dir):
        print(f"모든 파일이 '{output_dir}' 디렉토리에 저장되었습니다.")
    else:
        print("파일 저장에 실패했습니다.")
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    main()
