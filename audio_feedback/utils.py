import datetime
import os
import json
import math

def generate_analysis_id(video_id: str, analysis_type: str) -> str:
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    analysis_id = f"{video_id}_{analysis_type}_{current_timestamp}"
    return analysis_id

def save_feedback_to_json(feedback_data: dict, filename: str):
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
    if rms_value <= 0:
        return -120.0
    return 20 * math.log10(rms_value)

def get_stutter_words_at_timestamp(timestamp, word_timestamps):
    stutter_start = timestamp['start']
    stutter_end = timestamp['end']
    stutter_words = []
    for word_info in word_timestamps:
        word_start = word_info['start']
        word_end = word_info['end']
        if max(stutter_start, word_start) < min(stutter_end, word_end):
            stutter_words.append(word_info['word'].strip())
    return " ".join(stutter_words)

def find_full_sentence(stutter_words_text, full_transcript):
    if not stutter_words_text or not full_transcript:
        return ""
    try:
        start_index = full_transcript.find(stutter_words_text)
        if start_index == -1:
            return stutter_words_text
        sentence_start = 0
        for punc in ['.', '?', '!']:
            p_index = full_transcript.rfind(punc, 0, start_index)
            if p_index != -1 and p_index + 1 > sentence_start:
                sentence_start = p_index + 1
        sentence_end = len(full_transcript)
        for punc in ['.', '?', '!']:
            p_index = full_transcript.find(punc, start_index)
            if p_index != -1 and p_index < sentence_end:
                sentence_end = p_index + 1
        full_sentence = full_transcript[sentence_start:sentence_end].strip()
        return full_sentence
    except Exception:
        return stutter_words_text

def get_sentence_timestamps(word_timestamps, max_words=20):
    """
    단어 타임스탬프를 기반으로 문장 단위 타임스탬프를 생성합니다.
    (문장의 끝을 "." "?" "!"로 가정하거나, 최대 단어 수를 초과하면 분할)
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