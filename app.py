# unified_analysis_server.py (UNIFIED – single callback)
from flask import Flask, request, jsonify
import uuid
import os
import requests
import threading
import time
from tqdm import tqdm
import tempfile
import json

# 분석 모듈
import mainVideo
import audiomain

app = Flask(__name__)

analysis_status_map = {}
status_lock = threading.Lock()

def set_status(analysis_id, status):
    with status_lock:
        analysis_status_map[analysis_id] = status

def get_status(analysis_id):
    with status_lock:
        return analysis_status_map.get(analysis_id)

def notify_status(callback_url, payload, retries=1):
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            res = requests.post(callback_url, json=payload, timeout=60)
            print(f"[POST] {callback_url} -> {res.status_code}")
            if 200 <= res.status_code < 300:
                return True
        except Exception as e:
            print(f"[실패] POST (attempt {attempt}/{retries}) -> {e}")
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    return False

def build_callback_url(req):
    client_ip = req.headers.get("X-Forwarded-For", req.remote_addr)
    return f"http://{client_ip}:8080/analysis/callback"

def download_video(s3_url, output_path):
    try:
        # chunk_size: 다운로드/복사 시 한 번에 읽는 데이터 크기 (256KB)
        chunk_size = 1024 * 256
        if s3_url.startswith("file://"):
            local_path = s3_url.replace("file://", "")
            if os.name == "nt" and local_path.startswith("/") and ":" in local_path:
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
    except Exception as e:
        print(f"[다운로드 실패] {e}")
        return False

def process_analysis(s3_url: str, presentation_id: str, callback_url: str):
    now_tag = time.strftime('%Y%m%d%H%M%S')
    video_analysis_id = f"{presentation_id}_video_report_{now_tag}"
    audio_analysis_id = f"{presentation_id}_audio_report_{now_tag}"

    set_status(video_analysis_id, "IN_PROGRESS")
    set_status(audio_analysis_id, "IN_PROGRESS")

    video_results, audio_results = None, None
    video_err, audio_err = None, None

    with tempfile.TemporaryDirectory(prefix="dl_") as tmpdir:
        video_path = os.path.join(tmpdir, f"{presentation_id}.mp4")

        if not download_video(s3_url, video_path):
            payload = {
                "presentationId": presentation_id,
                "video": {
                    "analysisId": video_analysis_id,
                    "status": "FAILED",
                    "message": "download failed"
                },
                "audio": {
                    "analysisId": audio_analysis_id,
                    "status": "FAILED",
                    "message": "download failed"
                }
            }
            set_status(video_analysis_id, "FAILED")
            set_status(audio_analysis_id, "FAILED")
            notify_status(callback_url, payload)
            return False

        try:
            video_results = mainVideo.run(video_path)
            set_status(video_analysis_id, "COMPLETED")
        except Exception as e:
            video_err = str(e)
            set_status(video_analysis_id, "FAILED")

        try:
            audio_results = audiomain.amain(video_path, audio_analysis_id, presentation_id)
            set_status(audio_analysis_id, "COMPLETED")
        except Exception as e:
            audio_err = str(e)
            set_status(audio_analysis_id, "FAILED")

    payload = {
        "presentationId": presentation_id,
        "video": {
            "analysisId": video_analysis_id,
            "status": "FAILED" if video_err else "COMPLETED",
            **({"message": video_err} if video_err else {}),
            **({"results": video_results} if (video_err is None and isinstance(video_results, dict)) else {})
        },
        "audio": {
            "analysisId": audio_analysis_id,
            "status": "FAILED" if audio_err else "COMPLETED",
            **({"message": audio_err} if audio_err else {}),
            **({"results": audio_results} if (audio_err is None and isinstance(audio_results, dict)) else {})
        }
    }
    print('==========================================\n최종')
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    notify_status(callback_url, payload)
    return True

@app.route('/analysis', methods=['POST'])
def analyze_unified():
    data = request.get_json() or {}
    presentation_id = data.get("presentationId")
    s3_url = data.get("s3Url")
    if not all([presentation_id, s3_url]):
        return jsonify({"error": "presentationId, s3Url은 필수입니다."}), 400
    callback_url = build_callback_url(request)
    t = threading.Thread(
        target=process_analysis,
        args=(s3_url, presentation_id, callback_url),
        daemon=False
    )
    t.start()
    return jsonify({"presentationId": presentation_id, "status": "PENDING"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
