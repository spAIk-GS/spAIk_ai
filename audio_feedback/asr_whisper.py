# audio_feedback/asr_whisper.py
import whisper
import torch

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    return model

model = load_model()

def transcribe_audio(audio_path):
    # 단어별 타임스탬프를 얻기 위해 word_timestamps=True를 사용
    result = model.transcribe(audio_path, word_timestamps=True)
    text = result["text"]
    duration = result["segments"][-1]["end"] if result["segments"] else 0
    
    # 단어 타임스탬프 정보 추출
    word_timestamps = []
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                word_timestamps.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"]
                })
    
    # 수정된 반환값: 텍스트, 전체 길이, 단어별 타임스탬프
    return text, duration, word_timestamps