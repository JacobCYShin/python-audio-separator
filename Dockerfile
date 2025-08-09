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
RUN mkdir -p /tmp/audio-separator-models \
    && mkdir -p /tmp/output

# 빌드 타임 모델 로딩/실행 제거 (런타임에서 초기화)
# RUN LOCAL_TEST=true python3 handler.py

# 런포드 환경 변수 기본값
ENV RP_HANDLER_TIMEOUT=900 \
    RP_UPLOAD_ENABLE=true \
    RP_VERBOSE=true

# 헬스체크 (서버가 시작되었는지 간단 확인)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD pgrep -f "handler.py" || exit 1

# 포트 노출
EXPOSE 8000

# 핸들러 실행
CMD ["python3", "handler.py"]
