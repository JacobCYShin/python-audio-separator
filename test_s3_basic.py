#!/usr/bin/env python3
"""
S3 기본 기능 테스트 스크립트

이 스크립트는 AWS S3의 기본적인 업로드/다운로드 기능을 테스트합니다.
"""

import os
import boto3
import tempfile
from botocore.exceptions import ClientError, NoCredentialsError

# AWS S3 설정 (환경변수에서 가져오기)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'likebutter-bucket')

def test_s3_connection():
    """S3 연결을 테스트합니다."""
    print("=== S3 연결 테스트 ===")
    
    try:
        # S3 클라이언트 생성
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        # 버킷 이름
        bucket_name = S3_BUCKET_NAME
        
        # 버킷 존재 확인
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✓ 버킷 '{bucket_name}'에 연결 성공")
            return s3_client, bucket_name
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"✗ 버킷 '{bucket_name}'이 존재하지 않습니다")
            elif error_code == '403':
                print(f"✗ 버킷 '{bucket_name}'에 접근 권한이 없습니다")
            else:
                print(f"✗ 버킷 접근 오류: {error_code}")
            return None, None
            
    except NoCredentialsError:
        print("✗ AWS 인증 정보를 찾을 수 없습니다")
        print("환경변수를 설정해주세요:")
        print("export AWS_ACCESS_KEY_ID=your_access_key")
        print("export AWS_SECRET_ACCESS_KEY=your_secret_key")
        return None, None
    except Exception as e:
        print(f"✗ S3 연결 오류: {e}")
        return None, None

def test_s3_upload(s3_client, bucket_name):
    """S3 업로드를 테스트합니다."""
    print("\n=== S3 업로드 테스트 ===")
    
    try:
        # 테스트 파일 생성
        test_content = "이것은 S3 업로드 테스트 파일입니다.\n테스트 시간: " + str(os.popen('date').read().strip())
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        # S3 키 (파일 경로)
        s3_key = "test-files/test_upload.txt"
        
        print(f"로컬 파일: {temp_file_path}")
        print(f"S3 경로: s3://{bucket_name}/{s3_key}")
        
        # 파일 업로드
        s3_client.upload_file(temp_file_path, bucket_name, s3_key)
        print("✓ 파일 업로드 성공!")
        
        # 임시 파일 삭제
        os.unlink(temp_file_path)
        
        return s3_key
        
    except Exception as e:
        print(f"✗ 업로드 실패: {e}")
        return None

def test_s3_download(s3_client, bucket_name, s3_key):
    """S3 다운로드를 테스트합니다."""
    print("\n=== S3 다운로드 테스트 ===")
    
    try:
        # 다운로드할 로컬 파일 경로
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_file_path = f.name
        
        print(f"S3 경로: s3://{bucket_name}/{s3_key}")
        print(f"다운로드 경로: {temp_file_path}")
        
        # 파일 다운로드
        s3_client.download_file(bucket_name, s3_key, temp_file_path)
        print("✓ 파일 다운로드 성공!")
        
        # 다운로드된 파일 내용 확인
        with open(temp_file_path, 'r') as f:
            content = f.read()
            print(f"다운로드된 파일 내용:\n{content}")
        
        # 임시 파일 삭제
        os.unlink(temp_file_path)
        
        return True
        
    except Exception as e:
        print(f"✗ 다운로드 실패: {e}")
        return False

def test_s3_list_objects(s3_client, bucket_name):
    """S3 객체 목록을 조회합니다."""
    print("\n=== S3 객체 목록 조회 ===")
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
        
        if 'Contents' in response:
            print(f"버킷 '{bucket_name}'의 객체들:")
            for obj in response['Contents']:
                print(f"  - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print(f"버킷 '{bucket_name}'이 비어있습니다")
            
        return True
        
    except Exception as e:
        print(f"✗ 객체 목록 조회 실패: {e}")
        return False

def test_s3_delete_object(s3_client, bucket_name, s3_key):
    """S3 객체를 삭제합니다."""
    print("\n=== S3 객체 삭제 테스트 ===")
    
    try:
        print(f"삭제할 객체: s3://{bucket_name}/{s3_key}")
        
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        print("✓ 객체 삭제 성공!")
        
        return True
        
    except Exception as e:
        print(f"✗ 객체 삭제 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("S3 기본 기능 테스트 시작")
    print("=" * 50)
    
    # 환경변수 확인
    print("환경변수 확인:")
    env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'S3_BUCKET_NAME']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '***'
            print(f"  {var}: {masked_value}")
        else:
            print(f"  {var}: 설정되지 않음")
    
    # S3 연결 테스트
    s3_client, bucket_name = test_s3_connection()
    if not s3_client:
        print("\nS3 연결에 실패했습니다. 환경변수를 확인해주세요.")
        return
    
    # 업로드 테스트
    s3_key = test_s3_upload(s3_client, bucket_name)
    if not s3_key:
        print("업로드 테스트에 실패했습니다.")
        return
    
    # 다운로드 테스트
    download_success = test_s3_download(s3_client, bucket_name, s3_key)
    if not download_success:
        print("다운로드 테스트에 실패했습니다.")
        return
    
    # 객체 목록 조회 테스트
    test_s3_list_objects(s3_client, bucket_name)
    
    # 객체 삭제 테스트
    test_s3_delete_object(s3_client, bucket_name, s3_key)
    
    print("\n" + "=" * 50)
    print("모든 S3 기본 기능 테스트 완료!")

if __name__ == "__main__":
    main() 