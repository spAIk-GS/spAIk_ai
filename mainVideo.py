import cv2
import mediapipe as mp
import numpy as np
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine
import math
from tqdm import tqdm
import json
from collections import defaultdict

def infer_total_duration_sec(cap, total_frames, fps, timeline):
    if fps and fps > 0 and total_frames > 0:
        return float(total_frames) / float(fps)
    if timeline:
        return float(timeline[-1]["second"] + 1)
    return 0.0


def make_segments(timeline, segment_len_sec, total_duration_sec):
    body_segments, gaze_segments = [], []
    start = 0.0
    # 초별 빠른 접근을 위해 dict로
    by_sec = {t["second"]: t for t in timeline}
    while start < total_duration_sec:
        end = min(start + segment_len_sec, total_duration_sec)
        s0, s1 = int(start), int(end)  # 초 단위 포함 범위
        # 해당 구간의 초들 수집
        seg = [by_sec[s] for s in range(s0, s1) if s in by_sec]
        if seg:
            # 가중치: 초마다 샘플 수
            w = np.array([max(1, t["samples"]) for t in seg], dtype=float)

            # body: armMoveRatio
            arm = np.array([t["armMoveRatio"] for t in seg], dtype=float)
            arm_avg = float(np.average(arm, weights=w)) if arm.size else 0.0

            # gaze: gazeDownRatio
            gaze = np.array([t["gazeDownRatio"] for t in seg], dtype=float)
            gaze_avg = float(np.average(gaze, weights=w)) if gaze.size else 0.0

            body_segments.append({
                "start_time_sec": float(start),
                "end_time_sec": float(end),
                "value": round(arm_avg, 3)
            })
            gaze_segments.append({
                "start_time_sec": float(start),
                "end_time_sec": float(end),
                "value": 1 - round(gaze_avg, 3)
            })
        start = end
    return body_segments, gaze_segments


def run(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = None  # POS_MSEC 기반으로만 초 계산

    face_detector = FaceDetector("assets/face_detector.onnx")
    mark_detector = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    picked_frame = 0
    head_down = 0

    mp_pose = mp.solutions.pose
    pose_a = mp_pose.Pose()

    THRESHOLD = 0.3
    prev_landmarks = None
    frame_count = 0
    movement_detected = 0
    total_checked = 0

    # 진행률
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_steps = (total_frames // 15) if total_frames > 0 else None
    progress = tqdm(total=total_steps, desc="분석", unit="step", leave=True)

    # --- 초별 버킷 ---
    # sec_bins[sec] = dict(...)
    sec_bins = defaultdict(lambda: {
        "samples": 0,
        "face_frames": 0,
        "head_down_frames": 0,
        "arm_check_cnt": 0,
        "arm_move_cnt": 0,
        "pitch_sum": 0.0,
        "pitch_cnt": 0,
    })

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # 15프레임당 1회 분석(원 코드 유지)
            if frame_count % 15 != 0:
                continue

            # 진행률 업데이트
            if total_steps is not None:
                progress.update(1)

            # 현재 프레임 시간(밀리초) → 초 인덱스
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            sec = int(pos_msec / 1000.0) if pos_msec > 0 else (
                int((frame_count - 1) / fps) if fps else 0
            )

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_a.process(image_rgb)

            # Mediapipe 포즈 처리 및 팔 상대좌표
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                lw = np.array([lm[mp_pose.PoseLandmark.LEFT_WRIST].x,
                               lm[mp_pose.PoseLandmark.LEFT_WRIST].y,
                               lm[mp_pose.PoseLandmark.LEFT_WRIST].z])
                rw = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                               lm[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                               lm[mp_pose.PoseLandmark.RIGHT_WRIST].z])
                ls = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                               lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
                rs = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])

                rel_lw = lw - ls
                rel_rw = rw - rs

                shoulder_dist = np.linalg.norm(ls - rs)
                if shoulder_dist < 1e-6:
                    shoulder_dist = 1e-6

            # 얼굴/헤드포즈
            faces, _ = face_detector.detect(frame, 0.6)
            pitch_deg = None
            if len(faces) > 0:
                picked_frame += 1
                sec_bins[sec]["face_frames"] += 1

                face = refine(faces, frame_width, frame_height, 0.15)[0]
                x1, y1, x2, y2 = face[:4].astype(int)
                patch = frame[y1:y2, x1:x2]
                marks = mark_detector.detect([patch])[0].reshape([68, 2])
                marks *= (x2 - x1)
                marks[:, 0] += x1
                marks[:, 1] += y1
                pose_f = pose_estimator.solve(marks)
                rotation_matrix, _ = cv2.Rodrigues(pose_f[0])
                pitch_rad = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                pitch_deg = float(np.degrees(pitch_rad))

                # 초별 pitch 평균 집계
                sec_bins[sec]["pitch_sum"] += pitch_deg
                sec_bins[sec]["pitch_cnt"] += 1

                if pitch_deg < -18:
                    head_down += 1
                    sec_bins[sec]["head_down_frames"] += 1

            # 팔 움직임(임계 초과 비율)
            if results.pose_landmarks:
                if prev_landmarks is not None:
                    prev_rel_lw, prev_rel_rw = prev_landmarks
                    left_movement  = np.linalg.norm(rel_lw - prev_rel_lw) / shoulder_dist
                    right_movement = np.linalg.norm(rel_rw - prev_rel_rw) / shoulder_dist
                    avg_movement   = (left_movement + right_movement) / 2

                    total_checked += 1
                    sec_bins[sec]["arm_check_cnt"] += 1

                    if avg_movement > THRESHOLD:
                        movement_detected += 1
                        sec_bins[sec]["arm_move_cnt"] += 1

                prev_landmarks = [rel_lw, rel_rw]

            # 이 초에서 샘플한 횟수
            sec_bins[sec]["samples"] += 1

        # 진행률 마무리
        if total_steps is None:
            progress.set_description("분석(완료)")
        else:
            remaining = total_steps - progress.n
            if remaining > 0:
                progress.update(remaining)

        # 전체 요약
        head_down_ratio = head_down / picked_frame if picked_frame > 0 else 0.0
        arm_move_ratio  = (movement_detected / total_checked) if total_checked > 0 else 0.0


        # --- 초별 타임라인 계산 ---
        timeline = []
        for sec in sorted(sec_bins.keys()):
            b = sec_bins[sec]
            # 이 초에 얼굴이 잡힌 프레임 대비 고개 숙임 비율
            gaze_down_ratio = (b["head_down_frames"] / b["face_frames"]) if b["face_frames"] > 0 else 0.0
            # 이 초에 팔 움직임 감지 비율(임계 초과 비율)
            arm_ratio = (b["arm_move_cnt"] / b["arm_check_cnt"]) if b["arm_check_cnt"] > 0 else 0.0
            # 평균 pitch (deg)
            avg_pitch = (b["pitch_sum"] / b["pitch_cnt"]) if b["pitch_cnt"] > 0 else None

            timeline.append({
                "second": int(sec),
                "samples": int(b["samples"]),
                "faceFrames": int(b["face_frames"]),
                "gazeDownRatio": 1 - round(float(gaze_down_ratio), 3),
                "armMoveRatio": round(float(arm_ratio), 3),
                "avgPitchDeg": (round(float(avg_pitch), 3) if avg_pitch is not None else None)
            })
        
        SEGMENT_LEN = 45  # 초 단위 구간 길이

        # timeline 기반으로 구간 나누기
        max_sec = timeline[-1]["second"] if timeline else 0
        body_segments, gaze_segments = [], []

        for start in range(0, max_sec+1, SEGMENT_LEN):
            end = start + SEGMENT_LEN
            # 이 구간에 속한 타임라인 값 모으기
            seg_items = [t for t in timeline if start <= t["second"] < end]

            if seg_items:
            # body_movement(armMoveRatio 평균)
                arm_vals = [t["armMoveRatio"] for t in seg_items if t["armMoveRatio"] is not None]
                arm_avg = float(np.mean(arm_vals)) if arm_vals else 0.0
                body_segments.append({
                    "start_time_sec": float(start),
                    "end_time_sec": float(end),
                    "value": round(arm_avg, 3)
                })

                # gaze(gazeDownRatio 평균)
                gaze_vals = [t["gazeDownRatio"] for t in seg_items if t["gazeDownRatio"] is not None]
                gaze_avg = float(np.mean(gaze_vals)) if gaze_vals else 0.0
                gaze_segments.append({
                    "start_time_sec": float(start),
                    "end_time_sec": float(end),
                    "value": 1 - round(gaze_avg, 3)
                })

        total_duration_sec = infer_total_duration_sec(cap, total_frames, fps, timeline)
        body_segments, gaze_segments = make_segments(timeline, SEGMENT_LEN, total_duration_sec)

        from videoFG import generate_posture_feedback
        gaze_level, gesture_level = generate_posture_feedback(head_down_ratio, arm_move_ratio)

        report = {
            "body_movement": {
            "emotion": gesture_level,
            "value": round(arm_move_ratio, 3),
            "segments": body_segments
            },
            "gaze": {
                "emotion": gaze_level,
                "value": 1 - round(head_down_ratio, 3),
                "segments": gaze_segments
            }
            }
        return report

    finally:
        progress.close()
        cap.release()
        try:
            pose_a.close()
        except Exception:
            pass

if __name__ == "__main__":
    video_path = str(input("주소를 입력하세요: "))
    with open("data.json", "w", encoding="utf-8") as f:
        data = run(video_path)
        json.dump(data, f, ensure_ascii=False, indent=4)

        print("data2.json 저장 완료!")
    print("비디오 분석이 완료되었습니다.")
