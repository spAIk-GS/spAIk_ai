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
        "avg_rms": avg_rms,
        "rms_frames": rms_frames,
        "word_timestamps": word_timestamps
    }

def analyze_audio_segment(audio_path, start_time_sec, end_time_sec, word_timestamps):
    """
    Analyzes a specific segment of the audio file.
    Args:
        audio_path (str): Path to the audio file.
        start_time_sec (float): Start time of the segment in seconds.
        end_time_sec (float): End time of the segment in seconds.
        word_timestamps (list): List of all word timestamps from the full audio.
    Returns:
        dict: A dictionary containing analysis results for the segment.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=16000, offset=start_time_sec, duration=end_time_sec - start_time_sec)
    
    segment_duration = librosa.get_duration(y=y, sr=sr)
    
    # Transcript for the segment
    segment_words = [
        word for word in word_timestamps
        if start_time_sec <= word['start'] < end_time_sec
    ]
    segment_text = " ".join([word['word'] for word in segment_words])
    
    speaking_rate = calculate_speaking_rate(segment_text, segment_duration)

    # Pitch analysis for the segment
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    return {
        "start_time_sec": start_time_sec,
        "end_time_sec": end_time_sec,
        "speaking_rate_wpm": speaking_rate,
        "avg_pitch_hz": avg_pitch,
    }