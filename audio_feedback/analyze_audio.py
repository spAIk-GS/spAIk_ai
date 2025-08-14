# audio_feedback/analyze_audio.py
import librosa
import numpy as np
from audio_feedback.speaking_rate import calculate_speaking_rate
from audio_feedback.asr_whisper import transcribe_audio
import os
import subprocess

def analyze_audio_features(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    transcript, asr_duration, word_timestamps = transcribe_audio(audio_path)

    effective_duration = asr_duration if asr_duration > 0 else duration
    speaking_rate = calculate_speaking_rate(transcript, effective_duration)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    # 평균 RMS 값 대신 프레임별 RMS 값 전체를 반환합니다.
    rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    
    # 평균 RMS 값도 함께 반환하여 기존 기능 유지
    avg_rms = np.mean(rms_frames)

    return {
        "duration_sec": duration,
        "transcript": transcript,
        "speaking_rate_wpm": speaking_rate,
        "avg_pitch_hz": avg_pitch,
        "rms_frames": rms_frames,  # 프레임별 RMS 값
        "avg_rms": avg_rms,        # 평균 RMS 값
        "word_timestamps": word_timestamps
    }
