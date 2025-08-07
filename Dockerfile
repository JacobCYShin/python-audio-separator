# RunPod Serverless용 Audio Separator Dockerfile
FROM runpod/base:0.6.2-cuda12.1.0

# 시스템 패키지 업데이트 및 설치
RUN apt-get update && apt-get upgrade -y

# Python 명시적 설치
RUN apt-get install -y python3 python3-pip python3-venv

# 필수 패키지 설치
RUN apt-get install -y \
    ffmpeg \
    cuda-toolkit \
    cudnn9-cuda-12 \
    wget \
    curl

# Python 패키지 업데이트
RUN python3 -m pip install --upgrade pip

# Python 심볼릭 링크 생성 (python 명령어도 사용 가능하도록)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Python 경로 확인
RUN which python3 && python3 --version

# CUDA 12 호환 ONNX Runtime 설치
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 작업 디렉토리 설정
WORKDIR /workspace

# 필요한 파일들을 컨테이너에 복사
COPY . /workspace/

# Python 의존성 설치
RUN pip install -r requirements.txt

# 모델 캐시 디렉토리 생성
RUN mkdir -p /tmp/audio-separator-models
RUN mkdir -p /tmp/output

# 환경 변수 설정
ENV AUDIO_SEPARATOR_MODEL_DIR=/tmp/audio-separator-models
ENV PYTHONPATH=/workspace

# 모델 다운로드를 위한 테스트 실행 (빌드 시점)
RUN LOCAL_TEST=true python3 handler.py

# 포트 노출
EXPOSE 8000

# 핸들러 실행
CMD ["python3", "handler.py"]
