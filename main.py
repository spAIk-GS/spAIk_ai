import os
import time
import json
import numpy as np

from audio_feedback.extract_audio import extract_audio_from_video
from audio_feedback.analyze_audio import analyze_audio_features, analyze_audio_segment
from audio_feedback.stuttering_detector import detect_stuttering
from audio_feedback.feedback_generator import generate_audio_feedback
from audio_feedback.volume_detector import detect_volume_anomalies_by_sentence
from audio_feedback.utils import (
    generate_analysis_id,
    save_feedback_to_json,
    convert_rms_to_db,
    get_stutter_words_at_timestamp,
    find_full_sentence,
    get_sentence_timestamps,
    convert_numpy_to_python_types
)


def main():
    # 이 경로는 사용자 환경에 맞게 수정
    input_video_path = "C:/Users/SUNWOO/Desktop/spAIk_audio_ai-main/sample_input/mi3nu.mp4"
    extracted_audio_path = "C:/Users/SUNWOO/Desktop/spAIk/temp_output/sample_audio.wav"
    video_id = os.path.splitext(os.path.basename(input_video_path))[0]

    print("=== 1. 오디오 추출 중 ===")
    start = time.time()
    extract_audio_from_video(input_video_path, extracted_audio_path)
    end = time.time()
    print(f"[✓] 소요 시간: {end - start:.2f}초")

    print("=== 2. 오디오 분석 중 (전체) ===")
    start = time.time()
    features = analyze_audio_features(extracted_audio_path)
    end = time.time()
    print(f"[✓] 소요 시간: {end - start:.2f}초")
    
    total_duration = features['duration_sec']
    avg_rms = features['avg_rms']
    avg_rms_db = convert_rms_to_db(avg_rms)

    # 45초 단위 분석 결과를 저장할 리스트를 분리
    speed_segments = []
    pitch_segments = []
    segment_duration = 45 # 45초 단위로 변경
    
    print("=== 3. 오디오 분석 중 (45초 구간별) ===")
    start = time.time()
    # 45초 간격으로 반복
    for i in range(0, int(total_duration), segment_duration):
        segment_start = i
        segment_end = min(i + segment_duration, total_duration)
        if segment_end - segment_start > 0:
            segment_analysis = analyze_audio_segment(
                extracted_audio_path,
                segment_start,
                segment_end,
                features['word_timestamps']
            )
            # 말속도 세그먼트 데이터 저장
            speed_segments.append({
                "start_time_sec": round(float(segment_analysis.get("start_time_sec", 0)), 2),
                "end_time_sec": round(float(segment_analysis.get("end_time_sec", 0)), 2),
                "value": round(float(segment_analysis.get("speaking_rate_wpm", 0)), 2)
            })
            # 피치 세그먼트 데이터 저장
            pitch_segments.append({
                "start_time_sec": round(float(segment_analysis.get("start_time_sec", 0)), 2),
                "end_time_sec": round(float(segment_analysis.get("end_time_sec", 0)), 2),
                "value": round(float(segment_analysis.get("avg_pitch_hz", 0)), 2)
            })
    end = time.time()
    print(f"[✓] 소요 시간: {end - start:.2f}초")

    print("=== 4. 말더듬 감지 중 ===")
    start = time.time()
    stutter_results = detect_stuttering(extracted_audio_path)
    end = time.time()
    print(f"[✓] 소요 시간: {end - start:.2f}초")

    print("=== 5. 피드백 생성 중 ===")
    audio_feedback_results = generate_audio_feedback(features, avg_rms_db)
    
    volume_anomalies = detect_volume_anomalies_by_sentence(
        features['rms_frames'], 
        avg_rms_db, 
        sr=16000, 
        hop_length=512, 
        word_timestamps=features['word_timestamps']
    )
    
    sentences_with_timestamps = get_sentence_timestamps(features['word_timestamps'])
    
    stutter_count = stutter_results['stutter_count']
    stuttering_timestamps = stutter_results['stuttering_timestamps']
    stutter_feedback = stutter_results['stuttering_feedback']
    
    stutter_by_sentence = {}
    for timestamp in stuttering_timestamps:
        stutter_words = get_stutter_words_at_timestamp(timestamp, features['word_timestamps'])
        
        full_sentence = ""
        for sentence in sentences_with_timestamps:
            if timestamp['start'] >= sentence['start'] and timestamp['end'] <= sentence['end']:
                full_sentence = sentence['text']
                break
        
        if not full_sentence:
            full_sentence = find_full_sentence(stutter_words, features['transcript'])

        if full_sentence not in stutter_by_sentence:
            stutter_by_sentence[full_sentence] = {
                "timestamps": [],
                "stutter_words": []
            }
        
        stutter_by_sentence[full_sentence]["timestamps"].append(
            f"{timestamp['start']:.2f}s - {timestamp['end']:.2f}s"
        )
        stutter_by_sentence[full_sentence]["stutter_words"].append(stutter_words)
    
    stutter_details = []
    for sentence, details in stutter_by_sentence.items():
        stutter_details.append({
            "sentence": sentence,
            "timestamps": details["timestamps"],
            "stutter_words": details["stutter_words"]
        })
    
    final_feedback_report = {
        "analysisId": generate_analysis_id(video_id, "VoiceFeedback"),
        "videoId": video_id,
        "results": {
            "speed": {
                "feedback": audio_feedback_results.get("speed_feedback", ""),
                "value": round(float(audio_feedback_results.get("speaking_rate_wpm", 0.0)), 2),
                "level": audio_feedback_results.get("speed_level", ""),
                "segments": speed_segments
            },
            "pitch": {
                "feedback": audio_feedback_results.get("pitch_feedback", ""),
                "value": round(float(audio_feedback_results.get("avg_pitch_hz", 0.0)), 2),
                "level": audio_feedback_results.get("pitch_level", ""),
                "segments": pitch_segments
            },
            "volume": {
                "feedback": audio_feedback_results.get("volume_feedback", ""),
                "decibels": round(float(avg_rms_db), 2),
                "level": audio_feedback_results.get("volume_level", ""),
                "volume_anomalies": volume_anomalies
            },
            "stutter": {
                "feedback": stutter_feedback,
                "stutter_count": stutter_count,
                "stutter_details": stutter_details
            }
        }
    }

    # JSON 저장 전 NumPy 타입 변환
    final_feedback_report = convert_numpy_to_python_types(final_feedback_report)
    
    unique_id = generate_analysis_id(video_id, "full_report")
    json_filename = f"{unique_id}.json"
    save_feedback_to_json(final_feedback_report, json_filename)
    
    
if __name__ == "__main__":
    main()

