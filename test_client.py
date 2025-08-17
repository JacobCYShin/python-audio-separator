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

class AudioSeparatorClient:
    """Audio Separator API 클라이언트"""
    
    def __init__(self, base_url: str):
        """
        클라이언트 초기화
        
        Args:
            base_url: API 기본 URL (예: https://your-endpoint.runpod.net)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def get_models(self) -> Dict[str, Any]:
        """
        사용 가능한 모델 목록을 조회합니다.
        
        Returns:
            API 응답 데이터
        """
        try:
            response = self.session.get(f"{self.base_url}/api/models")
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
        오디오 파일을 분리합니다.
        
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
            request_data = {
                "audio_data": audio_data,
                "model_filename": model_filename,
                "output_format": output_format
            }
            
            if custom_output_names:
                request_data["custom_output_names"] = custom_output_names
            
            # API 호출
            response = self.session.post(
                f"{self.base_url}/api/separate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
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
            
            return True
            
        except Exception as e:
            print(f"파일 저장 실패: {e}")
            return False

def main():
    """메인 함수"""
    if len(sys.argv) < 3:
        print("사용법: python test_client.py <API_URL> <AUDIO_FILE>")
        print("예시: python test_client.py https://your-endpoint.runpod.net input.wav")
        sys.exit(1)
    
    api_url = sys.argv[1]
    audio_file = sys.argv[2]
    
    # 클라이언트 초기화
    client = AudioSeparatorClient(api_url)
    
    print("=== Audio Separator API 테스트 ===")
    print(f"API URL: {api_url}")
    print(f"오디오 파일: {audio_file}")
    print()
    
    # 1. 모델 목록 조회
    print("1. 모델 목록 조회 중...")
    models_response = client.get_models()
    
    if "error" in models_response:
        print(f"모델 목록 조회 실패: {models_response['error']}")
        sys.exit(1)
    
    print("사용 가능한 모델:")
    for model in models_response.get("models", [])[:5]:  # 처음 5개만 표시
        print(f"  - {model['model_filename']} ({model['friendly_name']})")
    print()
    
    # 2. 오디오 분리
    print("2. 오디오 분리 중...")
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
    print(f"사용된 모델: {separation_response.get('model_used', 'Unknown')}")
    print(f"출력 파일 수: {len(separation_response.get('output_files', {}))}")
    print()
    
    # 3. 결과 파일 저장
    print("3. 결과 파일 저장 중...")
    if client.save_output_files(separation_response, "output"):
        print("모든 파일이 'output' 디렉토리에 저장되었습니다.")
    else:
        print("파일 저장에 실패했습니다.")
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    main()
