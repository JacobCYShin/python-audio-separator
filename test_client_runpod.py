#!/usr/bin/env python3
"""
Audio Separator RunPod Serverless API 테스트 클라이언트

이 스크립트는 RunPod Serverless에 배포된 Audio Separator API를 테스트합니다.
- runsync(동기)와 run/status(비동기 폴링)를 모두 지원
- 서버 응답의 output 래핑을 해제하고, output_urls 또는 output_files를 저장
"""

import requests
import base64
import json
import os
import sys
from typing import Dict, Any, Optional
from urllib.parse import urlparse


class AudioSeparatorRunPodClient:
    """Audio Separator RunPod API 클라이언트"""

    def __init__(self, api_url: str, api_key: str = None):
        """
        클라이언트 초기화

        Args:
            api_url: RunPod Endpoint 기준 URL. 예시:
              - https://api.runpod.ai/v2/<ENDPOINT_ID>
              - 또는 기존 형식: https://api.runpod.ai/v2/<ENDPOINT_ID>/run, /runsync 중 하나
            api_key: RunPod API 키 (선택사항)
        """
        base = api_url.rstrip("/")
        if base.endswith("/run") or base.endswith("/runsync") or base.endswith("/status"):
            # 기존 형식에서 엔드포인트 베이스로 환원
            base = base.rsplit("/", 1)[0]
        self.base_url = base  # https://api.runpod.ai/v2/<ENDPOINT_ID>
        self.url_run = f"{self.base_url}/run"
        self.url_runsync = f"{self.base_url}/runsync"
        self.url_status_base = f"{self.base_url}/status"

        print(f"엔드포인트 BASE URL: {self.base_url}")
        print(f"RUN URL: {self.url_run}")
        print(f"RUNSYNC URL: {self.url_runsync}")

        self.api_key = api_key
        self.session = requests.Session()

        # 연결/재시도/타임아웃 설정
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

        # 5분 타임아웃 기본값
        self.session.timeout = 300

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers.update(headers)

    def _unwrap_output(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        """RunPod runsync/run 응답에서 output 래핑을 해제합니다."""
        if isinstance(response_json, dict) and "output" in response_json and isinstance(response_json["output"], dict):
            return response_json["output"]
        return response_json

    def _status_url(self, job_id: str) -> str:
        return f"{self.url_status_base}/{job_id}"

    def separate_audio(
        self,
        audio_file_path: str,
        output_format: str = "WAV",
        use_advanced: bool = False,
        return_type: str = "url",
        use_runsync: bool = True,
        poll_interval_sec: int = 5,
        max_wait_sec: int = 1800,
        model_filename: str = "Kim_Vocal_1.onnx",
    ) -> Dict[str, Any]:
        """
        오디오 분리를 수행합니다.

        Args:
            audio_file_path: 입력 오디오 파일 경로
            output_format: 출력 형식 (WAV/FLAC/...) 
            use_advanced: True면 4단계 고급 분리("advanced_separate"), False면 기본("separate")
            return_type: "url"(기본) 또는 "base64"
            use_runsync: True면 runsync 동기 처리, False면 run+status 폴링
            poll_interval_sec: 비동기 폴링 간격
            max_wait_sec: 비동기 최대 대기시간
            model_filename: 사용할 모델 파일명
        """
        # 파일 크기 안내
        file_size = os.path.getsize(audio_file_path)
        print(f"원본 파일 크기: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")

        # 파일 읽기 및 base64 인코딩
        with open(audio_file_path, "rb") as f:
            audio_data_b64 = base64.b64encode(f.read()).decode("utf-8")
        print(f"오디오(base64) 길이: {len(audio_data_b64)} chars")

        job_type = "advanced_separate" if use_advanced else "separate"
        payload = {
            "input": {
                "type": job_type,
                "audio_data": audio_data_b64,
                "output_format": output_format,
                "return_type": return_type,
                "model_filename": model_filename,
            }
        }

        if use_runsync:
            print("runsync 요청 전송...")
            resp = self.session.post(self.url_runsync, json=payload, timeout=self.session.timeout)
            print(f"runsync 상태: {resp.status_code}")
            try:
                resp_json = resp.json()
            except Exception:
                print(f"응답 텍스트: {resp.text}")
                resp.raise_for_status()
                return {"error": "Invalid JSON"}
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "details": resp_json}
            # RunPod 래핑 해제
            return self._unwrap_output(resp_json)
        else:
            print("run 비동기 제출...")
            submit = self.session.post(self.url_run, json=payload, timeout=self.session.timeout)
            print(f"run 상태: {submit.status_code}")
            submit.raise_for_status()
            submit_json = submit.json()
            job_id = submit_json.get("id")
            if not job_id:
                return {"error": "No job id returned", "details": submit_json}
            print(f"작업 ID: {job_id}")

            # /status 폴링
            waited = 0
            while waited < max_wait_sec:
                status_resp = self.session.get(self._status_url(job_id), timeout=self.session.timeout)
                if status_resp.status_code != 200:
                    print(f"status HTTP {status_resp.status_code}")
                try:
                    status_json = status_resp.json()
                except Exception:
                    print(f"status 응답 텍스트: {status_resp.text}")
                    return {"error": "Invalid status JSON"}

                status = status_json.get("status") or status_json.get("state")
                print(f"상태: {status}")
                if status == "COMPLETED":
                    return self._unwrap_output(status_json)
                if status == "FAILED":
                    return {"error": "Job failed", "details": status_json}

                import time
                time.sleep(poll_interval_sec)
                waited += poll_interval_sec

            return {"error": "Timeout waiting for job completion"}

    def save_outputs(self, response_data: Dict[str, Any], output_dir: str = ".") -> bool:
        """output_files(base64) 또는 output_urls(URL) 저장"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            saved_any = False

            # URL 저장
            if "output_urls" in response_data and isinstance(response_data["output_urls"], dict):
                for filename, url in response_data["output_urls"].items():
                    print(f"다운로드: {filename} <- {url}")
                    r = self.session.get(url, timeout=600)
                    r.raise_for_status()
                    path = os.path.join(output_dir, filename)
                    with open(path, "wb") as f:
                        f.write(r.content)
                    print(f"파일 저장됨: {path}")
                    saved_any = True

            # base64 저장
            if "output_files" in response_data and isinstance(response_data["output_files"], dict):
                for filename, b64data in response_data["output_files"].items():
                    path = os.path.join(output_dir, filename)
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(b64data))
                    print(f"파일 저장됨: {path}")
                    saved_any = True

            if not saved_any:
                print("저장할 출력이 없습니다. (output_urls/output_files 없음)")
                return False
            return True
        except Exception as e:
            print(f"파일 저장 실패: {e}")
            return False

    def test_connection(self) -> Dict[str, Any]:
        """서버 연결을 테스트합니다. runsync로 'list_models' 요청."""
        try:
            payload = {"input": {"type": "list_models"}}
            r = self.session.post(self.url_runsync, json=payload, timeout=30)
            try:
                j = r.json()
            except Exception:
                j = {"text": r.text}
            return {"status_code": r.status_code, "response": j}
        except Exception as e:
            return {"error": str(e)}


def main():
    """메인 함수"""
    if len(sys.argv) < 3:
        print("사용법: python test_client_runpod.py <API_BASE_OR_RUN_URL> <AUDIO_FILE> [API_KEY]")
        print("예시: python test_client_runpod.py https://api.runpod.ai/v2/<ENDPOINT_ID> input.wav")
        sys.exit(1)

    api_url = sys.argv[1]
    audio_file = sys.argv[2]
    api_key = sys.argv[3] if len(sys.argv) > 3 else None

    client = AudioSeparatorRunPodClient(api_url, api_key)

    print("서버 연결 테스트(runsync/list_models)...")
    ping = client.test_connection()
    print(json.dumps(ping, indent=2, ensure_ascii=False))
    if ping.get("status_code") != 200:
        print("연결 또는 권한 문제로 보입니다.")
        sys.exit(1)

    print("오디오 분리(runsync, 기본 분리, URL 반환)...")
    result = client.separate_audio(
        audio_file_path=audio_file,
        output_format="WAV",
        use_advanced=False,
        return_type="url",
        use_runsync=True,
    )

    print("응답:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("결과 파일 저장 중...")
    output_dir = "output_results"
    if client.save_outputs(result, output_dir):
        print(f"모든 파일이 '{output_dir}' 디렉토리에 저장되었습니다.")
    else:
        print("파일 저장 대상이 없습니다.")

    print("완료")


if __name__ == "__main__":
    main()
