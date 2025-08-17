#!/usr/bin/env python3
"""
boto3 설치 스크립트

이 스크립트는 S3 테스트에 필요한 boto3 라이브러리를 설치합니다.
"""

import subprocess
import sys

def install_boto3():
    """boto3를 설치합니다."""
    print("boto3 라이브러리 설치 중...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3>=1.26"])
        print("✓ boto3 설치 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ boto3 설치 실패: {e}")
        return False

def check_boto3():
    """boto3가 설치되어 있는지 확인합니다."""
    try:
        import boto3
        print(f"✓ boto3 버전: {boto3.__version__}")
        return True
    except ImportError:
        print("✗ boto3가 설치되어 있지 않습니다.")
        return False

if __name__ == "__main__":
    print("boto3 설치 스크립트")
    print("=" * 30)
    
    if check_boto3():
        print("boto3가 이미 설치되어 있습니다.")
    else:
        if install_boto3():
            check_boto3()
        else:
            print("수동으로 설치해주세요:")
            print("pip install boto3>=1.26")
    
    print("\n설치 완료 후 다음 명령어로 S3 테스트를 실행하세요:")
    print("python3 test_s3_basic.py") 