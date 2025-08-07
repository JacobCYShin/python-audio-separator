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
        
        # /run이 없으면 추가 (이미 포함된 경우 중복 방지)
        if not api_url.endswith('/run'):
            self.api_url = api_url + '/run'
        else:
            self.api_url = api_url
            
        print(f"최종 API URL: {self.api_url}")
            
        self.api_key = api_key
        self.session = requests.Session()
        
        # 연결 타임아웃 설정 (5분으로 증가)
        self.session.timeout = 300
        
        # 연결 재시도 설정
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        # 청크 전송을 위한 설정
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
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
            # 파일 크기 확인
            file_size = os.path.getsize(audio_file_path)
            print(f"원본 파일 크기: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
            # 파일이 너무 크면 경고
            if file_size > 10 * 1024 * 1024:  # 10MB
                print("⚠️  경고: 파일이 10MB를 초과합니다. 네트워크 연결이 불안정할 수 있습니다.")
            
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
            print(f"타임아웃 설정: {self.session.timeout}초")
            
            # 청크 전송 문제 해결을 위한 설정
            # 일반 JSON 전송 사용 (청크 전송 비활성화)
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Connection": "close"  # 연결을 명시적으로 닫음
            }
            
            print("API 요청 전송 중...")
            response = self.session.post(
                f"{self.api_url}", 
                json=payload,
                headers=headers,
                timeout=self.session.timeout
            )
            print(f"응답 상태 코드: {response.status_code}")
            print(f"응답 헤더: {dict(response.headers)}")
            
            # 응답 내용 확인
            try:
                response_json = response.json()
                print(f"응답 내용: {response_json}")
                
                # RunPod Serverless 비동기 처리 확인
                if 'id' in response_json and 'status' in response_json:
                    print(f"작업 ID: {response_json['id']}")
                    print(f"작업 상태: {response_json['status']}")
                    
                    # 작업이 완료될 때까지 기다리기
                    if response_json['status'] in ['IN_QUEUE', 'IN_PROGRESS']:
                        print("작업이 진행 중입니다. 동기식 응답을 기다리는 중...")
                        # 동기식 응답을 기다리기 위해 더 긴 타임아웃으로 재시도
                        return self._wait_for_sync_response(payload)
                        
            except Exception as e:
                print(f"응답 JSON 파싱 실패: {e}")
                print(f"응답 텍스트: {response.text}")
            
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
            print(f"연결 오류 타입: {type(e)}")
            if hasattr(e, 'args') and e.args:
                print(f"연결 오류 상세: {e.args}")
            return {"error": f"Connection error: {e}"}
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"청크 인코딩 오류: {e}")
            print("💡 해결 방법: 파일 크기를 줄이거나 네트워크 연결을 확인해주세요.")
            return {"error": f"Chunked encoding error: {e}"}
        except requests.exceptions.RequestException as e:
            print(f"고급 오디오 분리 실패: {e}")
            print(f"요청 예외 타입: {type(e)}")
            return {"error": str(e)}
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            print(f"오류 타입: {type(e)}")
            import traceback
            print(f"오류 상세: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def _wait_for_sync_response(self, payload: Dict[str, Any], max_wait_time: int = 600) -> Dict[str, Any]:
        """
        동기식 응답을 기다립니다.
        
        Args:
            payload: 원본 요청 데이터
            max_wait_time: 최대 대기 시간 (초)
            
        Returns:
            완료된 작업 결과
        """
        print(f"동기식 응답 대기 중... (최대 {max_wait_time}초)")
        
        # 더 긴 타임아웃으로 재시도
        extended_timeout = max_wait_time
        
        try:
            response = self.session.post(
                f"{self.api_url}", 
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=extended_timeout
            )
            
            print(f"동기식 응답 상태 코드: {response.status_code}")
            print(f"동기식 응답 헤더: {dict(response.headers)}")
            
            response.raise_for_status()
            result = response.json()
            print(f"동기식 응답 내용: {result}")
            return result
            
        except requests.exceptions.Timeout:
            print("동기식 응답 대기 시간 초과")
            return {"error": "Sync response timeout"}
        except Exception as e:
            print(f"동기식 응답 오류: {e}")
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

    def test_connection(self) -> Dict[str, Any]:
        """
        서버 연결을 테스트합니다.
        
        Returns:
            연결 테스트 결과
        """
        try:
            print(f"연결 테스트 중: {self.api_url}")
            
            # 간단한 ping 테스트
            test_payload = {
                "input": {
                    "type": "ping"
                }
            }
            
            print(f"ping 요청 전송: {test_payload}")
            response = self.session.post(self.api_url, json=test_payload, timeout=10)
            print(f"연결 테스트 응답: {response.status_code}")
            print(f"응답 헤더: {dict(response.headers)}")
            
            # 응답 내용 확인
            try:
                response_json = response.json()
                print(f"응답 내용: {response_json}")
            except Exception as e:
                print(f"응답 JSON 파싱 실패: {e}")
                print(f"응답 텍스트: {response.text}")
            
            response.raise_for_status()
            return {"success": True, "status_code": response.status_code, "response": response_json if 'response_json' in locals() else response.text}
            
        except requests.exceptions.ConnectionError as e:
            print(f"연결 테스트 실패 - 연결 오류: {e}")
            return {"error": f"Connection error: {e}"}
        except requests.exceptions.Timeout:
            print("연결 테스트 실패 - 타임아웃")
            return {"error": "Timeout"}
        except requests.exceptions.HTTPError as e:
            print(f"연결 테스트 실패 - HTTP 오류: {e}")
            print(f"응답 상태 코드: {e.response.status_code}")
            print(f"응답 내용: {e.response.text}")
            return {"error": f"HTTP error: {e}"}
        except Exception as e:
            print(f"연결 테스트 실패 - 기타 오류: {e}")
            import traceback
            print(f"오류 상세: {traceback.format_exc()}")
            return {"error": str(e)}

def main():
    """메인 함수"""
    if len(sys.argv) < 3:
        print("사용법: python test_client_runpod.py <API_URL> <AUDIO_FILE> [API_KEY]")
        print("예시: python test_client_runpod.py https://your-endpoint.runpod.net input.wav")
        print("예시: python test_client_runpod.py https://your-endpoint.runpod.net input.wav your-api-key")
        sys.exit(1)
    
    api_url = sys.argv[1]
    audio_file = sys.argv[2]
    api_key = sys.argv[3] if len(sys.argv) > 3 else None
    
    # 클라이언트 초기화
    client = AudioSeparatorRunPodClient(api_url, api_key)
    
    print("=== Audio Separator RunPod API 테스트 ===")
    print(f"API URL: {api_url}")
    print(f"오디오 파일: {audio_file}")
    print("고급 분리 모드 사용 (3단계 처리)")
    print()
    
    # 연결 테스트
    print("서버 연결 테스트 중...")
    connection_test = client.test_connection()
    if "error" in connection_test:
        print(f"연결 테스트 실패: {connection_test['error']}")
        print("API URL과 API 키를 확인해주세요.")
        sys.exit(1)
    else:
        print("연결 테스트 성공!")
    print()
    
    # 오디오 분리
    print("오디오 분리 중...")
    separation_response = client.advanced_separate_audio(audio_file)
    
    if "error" in separation_response:
        print(f"오디오 분리 실패: {separation_response['error']}")
        sys.exit(1)
    
    print("오디오 분리 완료!")
    print(f"완료된 단계: {separation_response.get('steps_completed', [])}")
    print(f"최종 출력: {separation_response.get('final_outputs', [])}")
    print(f"출력 파일 수: {len(separation_response.get('output_files', {}))}")
    print()
    
    # 결과 파일 저장
    print("결과 파일 저장 중...")
    output_dir = "output_advanced"
    if client.save_output_files(separation_response, output_dir):
        print(f"모든 파일이 '{output_dir}' 디렉토리에 저장되었습니다.")
    else:
        print("파일 저장에 실패했습니다.")
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    main()
