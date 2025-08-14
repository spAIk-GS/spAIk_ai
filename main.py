import os
import time
import json
import numpy as np

from audio_feedback.extract_audio import extract_audio_from_video
from audio_feedback.analyze_audio import analyze_audio_features
from audio_feedback.stuttering_detector import detect_stuttering
from audio_feedback.feedback_generator import generate_audio_feedback
from audio_feedback.volume_detector import detect_volume_anomalies_by_sentence
from audio_feedback.utils import (
    generate_analysis_id,
    save_feedback_to_json,
    convert_rms_to_db,
    get_stutter_words_at_timestamp,
    find_full_sentence,
    get_sentence_timestamps
)

def main():
    # 이 경로는 사용자 환경에 맞게 수정하세요.
    input_video_path = "C:/Users/SUNWOO/Desktop/spAIk_audio_ai-main/sample_input/mi3nu.mp4"
    extracted_audio_path = "C:/Users/SUNWOO/Desktop/spAIk/temp_output/sample_audio.wav"
    video_id = os.path.splitext(os.path.basename(input_video_path))[0]
    start_time = time.time()
    print("=== 1. 오디오 추출 중 ===")
    start = time.time()
    extract_audio_from_video(input_video_path, extracted_audio_path)
    end = time.time()
    print(f"[✓] 소요 시간: {end - start:.2f}초")

    print("=== 2. 오디오 분석 중 ===")
    start = time.time()
    features = analyze_audio_features(extracted_audio_path)
    end = time.time()
    print(f"[✓] 소요 시간: {end - start:.2f}초")
    
    avg_rms = features['avg_rms']
    avg_rms_db = convert_rms_to_db(avg_rms)

    print("=== 3. 말더듬 감지 중 ===")
    start = time.time()
    stutter_results = detect_stuttering(extracted_audio_path)
    end = time.time()
    print(f"[✓] 소요 시간: {end - start:.2f}초")

    print("=== 4. 피드백 생성 중 ===")
    audio_feedback_results = generate_audio_feedback(features, avg_rms_db)
    
    # 문장별 볼륨 이상 감지 (문장 길이 제한이 여기서는 이미 적용됩니다.)
    volume_anomalies = detect_volume_anomalies_by_sentence(
        features['rms_frames'], 
        avg_rms_db, 
        sr=16000, 
        hop_length=512, 
        word_timestamps=features['word_timestamps']
    )
    
    # 새로운 로직: 전체 대본을 짧은 문장들로 분할합니다.
    sentences_with_timestamps = get_sentence_timestamps(features['word_timestamps'])
    
    stutter_count = stutter_results['stutter_count']
    stuttering_timestamps = stutter_results['stuttering_timestamps']
    stutter_feedback = stutter_results['stuttering_feedback']
    
    stutter_by_sentence = {}
    for timestamp in stuttering_timestamps:
        stutter_words = get_stutter_words_at_timestamp(timestamp, features['word_timestamps'])
        
        # 수정된 로직: 짧게 분할된 문장 리스트에서 말더듬이 포함된 문장을 찾습니다.
        full_sentence = ""
        for sentence in sentences_with_timestamps:
            if timestamp['start'] >= sentence['start'] and timestamp['end'] <= sentence['end']:
                full_sentence = sentence['text']
                break
        
        if not full_sentence:
            # 적절한 문장을 찾지 못하면 기존 find_full_sentence 로직을 사용합니다.
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
                "value": round(float(audio_feedback_results.get("speaking_rate_wpm", 0.0)), 2)
            },
            "pitch": {
                "feedback": audio_feedback_results.get("pitch_feedback", ""),
                "value": round(float(audio_feedback_results.get("avg_pitch_hz", 0.0)), 2)
            },
            "volume": {
                "feedback": audio_feedback_results.get("volume_feedback", ""),
                "decibels": round(float(avg_rms_db), 2),
                "volume_anomalies": volume_anomalies
            },
            "stutter": {
                "feedback": stutter_feedback,
                "stutter_count": stutter_count,
                "stutter_details": stutter_details
            }
        }
    }

    unique_id = generate_analysis_id(video_id, "full_report")
    json_filename = f"{unique_id}.json"
    save_feedback_to_json(final_feedback_report, json_filename)
    end_time = time.time()
    print(f"총 소요시간 {end_time - start_time}초")
   
if __name__ == "__main__":
    main()




