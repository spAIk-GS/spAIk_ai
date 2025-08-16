import datetime
import os
import json
import math
import numpy as np

def generate_analysis_id(video_id: str, analysis_type: str) -> str:
    """
    영상 ID와 분석 타입을 기반으로 고유한 분석 ID를 생성합니다.
    """
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    analysis_id = f"{video_id}_{analysis_type}_{current_timestamp}"
    return analysis_id

def save_feedback_to_json(feedback_data: dict, filename: str):
    """
    피드백 데이터를 JSON 파일로 저장합니다.
    """
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=4)
        print(f"피드백이 성공적으로 '{file_path}'에 저장되었습니다.")
    except IOError as e:
        print(f"파일 저장 중 오류가 발생했습니다: {e}")

def convert_rms_to_db(rms_value):
    """
    RMS 값을 데시벨(dB)로 변환합니다.
    """
    if rms_value <= 0:
        return -120.0
    return 20 * math.log10(rms_value)

def get_stutter_words_at_timestamp(timestamp, word_timestamps):
    """
    주어진 타임스탬프에 해당하는 말더듬 단어들을 찾습니다.
    """
    stutter_start = timestamp['start']
    stutter_end = timestamp['end']
    stutter_words = []
    for word_info in word_timestamps:
        if word_info['start'] >= stutter_start and word_info['end'] <= stutter_end:
            stutter_words.append(word_info['word'])
    return " ".join(stutter_words)

def find_full_sentence(stutter_words_text, full_transcript):
    """
    말더듬 단어가 포함된 전체 문장을 찾아 반환합니다.
    """
    try:
        # 말더듬 단어가 속한 문장을 찾기 위해 텍스트를 문장 단위로 분리합니다.
        sentences = full_transcript.split('.')
        for sentence in sentences:
            if stutter_words_text.strip() in sentence:
                return sentence.strip() + '.'
        return stutter_words_text
    except Exception:
        return stutter_words_text

def get_sentence_timestamps(word_timestamps, max_words=20):
    """
    단어 타임스탬프를 기반으로 문장 단위 타임스탬프를 생성합니다.
    """
    sentences = []
    current_sentence_words = []
    
    for i, word_info in enumerate(word_timestamps):
        current_sentence_words.append(word_info)
        
        is_end_of_sentence = any(p in word_info['word'] for p in ['.', '?', '!', '...'])
        is_max_length_reached = len(current_sentence_words) >= max_words
        
        if is_end_of_sentence or is_max_length_reached:
            start_time = current_sentence_words[0]['start']
            end_time = current_sentence_words[-1]['end']
            sentence_text = " ".join([w['word'] for w in current_sentence_words])
            sentences.append({
                'text': sentence_text.strip(),
                'start': start_time,
                'end': end_time
            })
            current_sentence_words = []

    if current_sentence_words:
        start_time = current_sentence_words[0]['start']
        end_time = current_sentence_words[-1]['end']
        sentence_text = " ".join([w['word'] for w in current_sentence_words])
        sentences.append({
            'text': sentence_text.strip(),
            'start': start_time,
            'end': end_time
        })

    return sentences

def convert_numpy_to_python_types(obj):
    """
    재귀적으로 딕셔너리나 리스트 내부의 numpy.float32 객체를
    Python의 표준 float으로 변환하는 헬퍼 함수
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python_types(i) for i in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj