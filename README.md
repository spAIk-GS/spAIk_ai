spAIk: AI 발표 & 면접 피드백 시스템
**spAIk**는 사용자가 업로드한 영상 파일을 분석하여, 발표 및 면접 역량을 향상시킬 수 있는 종합적인 피드백을 제공하는 AI 시스템입니다. 음성 및 영상 데이터를 정량적, 정성적으로 분석해 사용자에게 맞춤형 리포트를 제공합니다.

🚀 주요 기능
음성 분석:

STT (Speech-to-Text): Whisper를 활용해 음성을 텍스트로 변환합니다.

음성 지표 분석: librosa를 사용해 말속도(WPM), 피치(Hz), 음량(dB)을 추출하고, 45초 구간별 분석 데이터를 제공합니다.

말더듬 감지: 오디오의 낮은 에너지 레벨을 기준으로 멈칫거림이나 침묵 구간을 감지하여 횟수와 위치를 분석합니다.

영상 분석:

비언어적 요소 분석: MediaPipe, OpenCV, ONNX Runtime 기반의 Head-Pose-Estimation 모델을 활용하여 시선 유지 비율 및 제스처 점수를 평가합니다.

피드백 생성: 분석된 모든 데이터를 기반으로 논리성, 명확성, 구성에 대한 종합적인 피드백을 제공합니다.

데이터 출력:분석 결과는 JSON 형식으로 구조화되어 저장되며, 음성 및 영상 피드백을 구분하여 제공합니다.

🛠️ 기술 스택
언어: Python 3.11

주요 라이브러리: librosa, ffmpeg, whisper, numpy, mediapipe, opencv-python, onnxruntime


환경: dotenv (환경 변수 관리)

📁 프로젝트 구조
```
spAIk_ai/
├── audio_feedback/
│   ├── ai_feedback.py          # Gemini API를 활용한 텍스트 피드백 생성
│   ├── analyze_audio.py        # 음성 특징(피치, 속도 등) 분석
│   ├── asr_whisper.py          # STT 변환
│   ├── extract_audio.py        # 영상에서 오디오 추출
│   ├── feedback_generator.py   # 분석 지표 기반 피드백 생성
│   ├── speaking_rate.py        # 말속도 관련 유틸리티
│   ├── stuttering_detector.py  # 말더듬 감지
│   ├── utils.py                # 공통 유틸리티 함수 모음
│   └── volume_detector.py      # 음량 이상 감지
└── main.py                     # 메인 실행 파일
```
⚙️ 설치 및 실행 방법

사전 준비

Python 3.11 이상이 설치되어 있어야 합니다.

ffmpeg가 시스템에 설치되어 있어야 합니다. (예: brew install ffmpeg 또는 공식 웹사이트 참고)

환경 설정
레포지토리를 클론합니다.
```
git clone https://github.com/spAIk-GS/spAIk_ai.git
cd spAIk_ai
```
가상 환경을 생성하고 활성화합니다.
```
python -m venv venv
venv\Scripts\activate
source venv/bin/activate
```
필요한 라이브러리를 설치합니다.
```
pip install -r requirements.txt
```


실행
main.py 파일을 실행하여 분석을 시작합니다. 주의: main.py 파일의 input_video_path 변수 값을 분석할 영상 파일 경로로 변경해야 합니다.
```
python main.py
```

