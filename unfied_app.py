# unified_analysis_server.py
from flask import Flask, request, jsonify
import uuid
import os
import requests
import threading
import time
from tqdm import tqdm
import tempfile
import traceback

# 분석 모듈
import mainVideo          # mainVideo.run(video_path) -> dict
import audiomain          # audiomain.amain(video_path, analysis_id, presentation_id) -> dict

app = Flask(__name__)

# =========================
# 상태 관리 (공통)
# =========================
analysis_status_map = {}
status_lock = threading.Lock()

def set_status(analysis_id, status):
    with status_lock:
        analysis_status_map[analysis_id] = status

def get_status(analysis_id):
    with status_lock:
        return analysis_status_map.get(analysis_id)

# =========================
# 네트워크 유틸 (공통)
# =========================
def notify_status(callback_url, payload, retries=3):
    """콜백 POST (payload는 이미 완성된 dict) - 지수 백오프"""
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            res = requests.post(callback_url, json=payload, timeout=10)
            print(f"[POST] {callback_url} -> {res.status_code}")
            if 200 <= res.status_code < 300:
                return True
        except Exception as e:
            print(f"[실패] POST (attempt {attempt}/{retries}) -> {e}")
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    return False

def build_callback_url(req, kind: str):
    """
    요청 보낸 클라이언트 IP 기준 콜백 URL 구성.
    kind: "video" | "audio"
    콜백 서버는 클라이언트 측 8080에 열려있다는 전제를 유지.
    """
    client_ip = req.headers.get("X-Forwarded-For", req.remote_addr)
    return f"http://{client_ip}:8080/analysis/callback/{kind}"

# =========================
# 다운로드 (공통)
# =========================
def download_video(s3_url, output_path):
    """presigned S3 URL 또는 file:// 로 영상 다운로드 (tqdm 진행률)"""
    try:
        chunk_size = 1024 * 256  # 256KB

        if s3_url.startswith("file://"):
            # 로컬 파일 복사
            local_path = s3_url.replace("file://", "")
            if os.name == "nt" and local_path.startswith("/") and ":" in local_path:
                # Windows file://C:/... 형태 보정
                local_path = local_path[1:]

            total_size = os.path.getsize(local_path)
            print(f"[복사] {local_path} -> {output_path}")
            with open(local_path, "rb") as src, open(output_path, "wb") as dst, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="다운로드(로컬 복사)", leave=True
            ) as pbar:
                while True:
                    buf = src.read(chunk_size)
                    if not buf:
                        break
                    dst.write(buf)
                    pbar.update(len(buf))
            print("[다운로드 완료] (로컬 복사)")
            return True

        # 원격 다운로드
        with requests.get(s3_url, stream=True, timeout=10) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(output_path, 'wb') as f, tqdm(
                total=total_size if total_size > 0 else None,
                unit="B", unit_scale=True, desc="다운로드", leave=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("[다운로드 완료]")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[다운로드 실패] {e}")
        return False
    except Exception as e:
        print(f"[다운로드 예외] {e}")
        return False

# =========================
# 작업 실행기 (비디오)
# =========================
def process_video(s3_url, analysis_id, presentation_id, callback_url):
    set_status(analysis_id, "IN_PROGRESS")

    with tempfile.TemporaryDirectory(prefix="dl_") as tmpdir:
        video_path = os.path.join(tmpdir, f"{analysis_id}.mp4")

        # 1) 다운로드
        if not download_video(s3_url, video_path):
            set_status(analysis_id, "FAILED")
            fail_payload = {
                "analysisId": analysis_id,
                "videoId": presentation_id,
                "status": "FAILED",
                "message": "Download failed"
            }
            notify_status(callback_url, fail_payload)
            return

        try:
            # 2) 분석 실행 (mainVideo.run이 dict 반환)
            result_data = mainVideo.run(video_path)
            if not isinstance(result_data, dict):
                raise ValueError("mainVideo.run 결과 형식이 dict가 아닙니다.")

            final_payload = {
                "analysisId": analysis_id,
                "videoId": presentation_id,
                "result": result_data
            }
            print(final_payload)

            set_status(analysis_id, "COMPLETED")

            # 선택: 요약 디버그 출력
            summary = final_payload.get("result", {}).get("content_summary")
            if summary is not None:
                print("\n[DEBUG] content_summary =======================")
                print(summary)
                print("==============================================\n")
            else:
                print("[DEBUG] content_summary 없음")

            notify_status(callback_url, final_payload)

        except Exception as e:
            err_trace = traceback.format_exc()
            print(f"[분석 실패] {e}")
            print(err_trace)

            set_status(analysis_id, "FAILED")
            fail_payload = {
                "analysisId": analysis_id,
                "videoId": presentation_id,
                "message": f"{str(e)}\n\n{err_trace}"[:4000]
            }
            notify_status(callback_url, fail_payload)
            return

# =========================
# 작업 실행기 (오디오)
# =========================
def process_audio(s3_url, analysis_id, presentation_id, callback_url):
    set_status(analysis_id, "IN_PROGRESS")

    with tempfile.TemporaryDirectory(prefix="dl_") as tmpdir:
        video_path = os.path.join(tmpdir, f"{analysis_id}.mp4")

        # 1) 다운로드
        if not download_video(s3_url, video_path):
            set_status(analysis_id, "FAILED")
            fail_payload = {
                "analysisId": analysis_id,
                "videoId": presentation_id, # status는 빼는 조건
                "status": "FAILED",
            }
            notify_status(callback_url, fail_payload)
            return

        try:
            # 2) 분석 실행 (audiomain.amain이 dict 반환)
            result_data = audiomain.amain(video_path, analysis_id, presentation_id)
            if not isinstance(result_data, dict):
                raise ValueError("audiomain.amain 결과 형식이 dict가 아닙니다.")

            final_payload = {
                "analysisId": analysis_id,
                "videoId": presentation_id,
                "status": "COMPLETED",
                "result": result_data
            }
            set_status(analysis_id, "COMPLETED")
            notify_status(callback_url, final_payload)

        except Exception as e:
            err_trace = traceback.format_exc()
            print(f"[분석 실패] {e}")
            print(err_trace)

            set_status(analysis_id, "FAILED")
            fail_payload = {
                "analysisId": analysis_id,
                "videoId": presentation_id,
                "status": "FAILED",
                "message": f"{str(e)}\n\n{err_trace}"[:4000]
            }
            notify_status(callback_url, fail_payload)
            return

# =========================
# 엔드포인트
# =========================
@app.route('/analysis/video', methods=['POST'])
def analyze_video():
    data = request.get_json()
    presentation_id = data.get("presentationId")
    s3_url = data.get("s3Url")

    if not all([presentation_id, s3_url]):
        return jsonify({"error": "presentationId, s3Url은 필수입니다."}), 400

    analysis_id = f"video-analysis-uuid-{uuid.uuid4()}"
    set_status(analysis_id, "PENDING")

    callback_url = build_callback_url(request, "video")

    t = threading.Thread(
        target=process_video,
        args=(s3_url, analysis_id, presentation_id, callback_url),
        daemon=False
    )
    t.start()

    return jsonify({"analysisId": analysis_id, "status": "PENDING"})

@app.route('/analysis/audio', methods=['POST'])
def analyze_audio():
    data = request.get_json()
    presentation_id = data.get("presentationId")
    s3_url = data.get("s3Url")

    if not all([presentation_id, s3_url]):
        return jsonify({"error": "presentationId, s3Url은 필수입니다."}), 400

    analysis_id = f"audio-analysis-uuid-{uuid.uuid4()}"
    set_status(analysis_id, "PENDING")

    callback_url = build_callback_url(request, "audio")

    t = threading.Thread(
        target=process_audio,
        args=(s3_url, analysis_id, presentation_id, callback_url),
        daemon=False
    )
    t.start()

    return jsonify({"analysisId": analysis_id, "status": "PENDING"})

if __name__ == '__main__':
    # 하나의 서버로 통합: 0.0.0.0:5000
    app.run(host='0.0.0.0', port=5000)
