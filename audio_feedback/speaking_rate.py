# audio_feedback/speaking_rate.py
import numpy as np
from audio_feedback.utils import get_sentence_timestamps


def calculate_speaking_rate(asr_text: str, duration_sec: float) -> float:
    """
    ASR 텍스트와 음성 길이(초)를 받아 말속도(분당 단어 수)를 계산
    """
    word_count = len(asr_text.strip().split())
    if duration_sec <= 0:
        return 0.0
    wpm = (word_count / duration_sec) * 60
    return wpm

def detect_volume_anomalies_by_sentence(rms_frames, avg_rms_db, sr=16000, hop_length=512, word_timestamps=None):
    """
    문장별 평균 RMS 값을 분석하여 음량이 정상 범주를 벗어나는 문장을 감지합니다.
    평균 음량을 기준으로 동적 임계값을 설정합니다.
    """
    if word_timestamps is None or not word_timestamps:
        return []
        
    # 영상 전체 평균 dB을 기준으로 동적 임계값 설정
    # 예: 평균 dB에서 4dB 낮으면 '너무 작음', 4dB 높으면 '너무 큼'으로 간주
    quiet_db_threshold = avg_rms_db - 4.0
    loud_db_threshold = avg_rms_db + 4.0

    frame_rate = sr / hop_length
    sentences = get_sentence_timestamps(word_timestamps)
    anomalies = []

    for sentence in sentences:
        start_frame = int(sentence['start'] * frame_rate)
        end_frame = int(sentence['end'] * frame_rate)
        
        sentence_rms_frames = rms_frames[start_frame:end_frame]
        
        if len(sentence_rms_frames) > 0:
            avg_rms_sentence = np.mean(sentence_rms_frames)
            avg_db_sentence = convert_rms_to_db(avg_rms_sentence)

            feedback_message = ""
            if avg_db_sentence < quiet_db_threshold:
                feedback_message = "음량이 너무 작습니다."
            elif avg_db_sentence > loud_db_threshold:
                feedback_message = "음량이 너무 큽니다."
            
            if feedback_message:
                anomalies.append({
                    "sentence": sentence['text'],
                    "timestamp": f"{sentence['start']:.2f}s - {sentence['end']:.2f}s",
                    "avg_decibels": round(float(avg_db_sentence), 2),
                    "feedback": feedback_message
                })
    
    return anomalies

def convert_rms_to_db(rms_value):
    if rms_value <= 0:
        return -120.0
    return 20 * math.log10(rms_value)