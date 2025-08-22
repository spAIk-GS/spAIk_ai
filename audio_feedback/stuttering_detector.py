import librosa
import numpy as np

def detect_stuttering(audio_path, frame_length=2048, hop_length=512, threshold=0.008):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except FileNotFoundError:
        return {
            "stutter_count": 0,
            "stuttering_timestamps": [],
            "stuttering_feedback": "오디오 파일을 찾을 수 없어 말더듬 분석을 수행할 수 없습니다."
        }
    except Exception as e:
        return {
            "stutter_count": 0,
            "stuttering_timestamps": [],
            "stuttering_feedback": f"오디오 처리 중 오류 발생: {e}"
        }

    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    stutter_frames = energy < threshold
    
    stuttering_timestamps = []
    in_stutter = False
    stutter_start_frame = -1
    
    for i in range(len(stutter_frames)):
        if stutter_frames[i] and not in_stutter:
            in_stutter = True
            stutter_start_frame = i
        elif not stutter_frames[i] and in_stutter:
            in_stutter = False
            stutter_end_frame = i - 1
            
            start_time = librosa.frames_to_time(stutter_start_frame, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(stutter_end_frame, sr=sr, hop_length=hop_length)
            
            if end_time - start_time > 1:
                stuttering_timestamps.append({"start": start_time, "end": end_time})

    if in_stutter:
        start_time = librosa.frames_to_time(stutter_start_frame, sr=sr, hop_length=hop_length)
        end_time = librosa.get_duration(y=y, sr=sr)
        if end_time - start_time > 1:
            stuttering_timestamps.append({"start": start_time, "end": end_time})
    
    count = len(stuttering_timestamps)
    duration_seconds = librosa.get_duration(y=y, sr=sr)
    stuttering_feedback_message = get_stuttering_feedback(count, duration_seconds)

    return {
        "stutter_count": count,
        "stuttering_timestamps": stuttering_timestamps,
        "stuttering_feedback": stuttering_feedback_message
    }

def get_stuttering_feedback(stuttering_counts, total_duration_seconds):
    if total_duration_seconds <= 0:
        return "영상 길이가 짧아 말더듬 횟수를 정확히 평가하기 어렵습니다."

    total_duration_minutes = total_duration_seconds / 60
    
    if total_duration_minutes < 0.1:
        return f"영상 길이가 짧아 (약 {total_duration_seconds:.1f}초) 말더듬 횟수({stuttering_counts}회)를 분당 기준으로 평가하기 어렵습니다."

    stuttering_per_minute = stuttering_counts / total_duration_minutes

    if stuttering_counts == 0:
        return "✅ 말씀하시는 동안 더듬거나 멈칫거림이 전혀 없었습니다. 매우 안정적인 발화였습니다."
    elif stuttering_per_minute < 0.5:
        return f"💬 말씀하시는 동안 더듬거나 멈칫거림이 거의 없었습니다. 발화가 매우 자연스러웠습니다."
    elif stuttering_per_minute < 2:
        return f"⚠️ 말씀하시는 동안 가끔 더듬거나 멈칫거림이 있었습니다. 조금 더 침착하게 발화 속도를 조절해 보세요."
    else:
        return f"❌ 말씀하시는 동안 더듬거나 멈칫거림이 다소 많았습니다. 긴장을 풀고 천천히 말하는 연습이 필요합니다. 답변 내용을 미리 정리하면 도움이 됩니다."
