# **spAIk: AI 발표 피드백 시스템**

#### spAIk는 사용자가 업로드한 영상 파일을 분석하여, 발표 역량을 향상시킬 수 있는 종합적인 피드백을 제공하는 AI 시스템입니다. 
#### 음성 및 영상 데이터를 정량적, 정성적으로 분석해 사용자에게 맞춤형 리포트를 제공합니다.

### 🚀 주요 기능

##### 음성 분석: STT (Speech-to-Text): Whisper를 활용해 음성을 텍스트로 변환합니다.

##### 음성 지표 분석: librosa를 사용해 말속도(WPM), 피치(Hz), 음량(dB)을 추출하고, 45초 구간별 분석 데이터를 제공합니다.

##### 말더듬 감지: 오디오의 낮은 에너지 레벨을 기준으로 멈칫거림이나 침묵 구간을 감지하여 횟수와 위치를 분석합니다.

##### 영상 분석: 비언어적 요소 분석: MediaPipe, OpenCV, ONNX Runtime 기반의 Head-Pose-Estimation 모델을 활용하여 시선 유지 비율 및 제스처 점수를 평가합니다.

##### 피드백 생성: 분석된 모든 데이터를 기반으로 논리성, 명확성, 구성에 대한 종합적인 피드백을 제공합니다.

##### 데이터 출력:분석 결과는 JSON 형식으로 구조화되어 저장되며, 음성 및 영상 피드백을 구분하여 제공합니다.


### 🛠️ 기술 스택
##### 언어: Python 3.10.13

##### 주요 라이브러리: librosa, ffmpeg, whisper, numpy, mediapipe, opencv-python, onnxruntime

### 📁 프로젝트 구조
```
spAIk_ai/
├── audio_feedback/
│   ├── analyze_audio.py        # 음성 특징(피치, 속도 등) 분석
│   ├── asr_whisper.py          # STT 변환
│   ├── extract_audio.py        # 영상에서 오디오 추출
│   ├── feedback_generator.py   # 분석 지표 기반 피드백 생성
│   ├── speaking_rate.py        # 말속도 관련 유틸리티
│   ├── stuttering_detector.py  # 말더듬 감지
│   ├── utils.py                # 공통 유틸리티 함수 모음
│   └── volume_detector.py      # 음량 이상 감지
│
├── video_feedback/               # 영상 관련 피드백 모듈
│   ├── assets/                   # 얼굴 감지 모델 파일(.onnx 등)
│   ├── face_detection.py         # 얼굴 감지 클래스
│   ├── mark_detection.py         # 얼굴 랜드마크 감지 클래스
│   ├── pose_estimation.py        # 머리 자세 추정 클래스
│   ├── utils.py                  # 공통 유틸리티 함수 모음
│   └── videoFG.py                # 평가 기준/추가 기능 모듈
├── app.py                      # 서버 진입점
├── videomain.py                # 비디오 분석 메인 실행 파일
└── audiomain.py                # 오디오 분석 메인 실행 파일
```
### ⚙️ 설치 및 실행 방법

##### 1. 사전 준비

##### Python 3.10.13 이 설치되어 있어야 합니다.

##### ffmpeg 프로그램이 라이브러리 외 추가로 시스템에 설치되어 있어야 합니다. (예: brew install ffmpeg 또는 공식 웹사이트 참고)

##### 2. 환경 설정
##### 레포지토리를 클론합니다.
```
git clone https://github.com/spAIk-GS/spAIk_ai.git
cd spAIk_ai
```
##### 가상 환경을 생성하고 활성화합니다.
```
python -m venv venv
venv\Scripts\activate
source venv/bin/activate
```
##### 필요한 라이브러리를 설치합니다.
```
pip install -r requirements.txt
```


##### 3. 실행
 ##### 오디오 분석은 audiomain.py 파일 input_video_path 인자에 영상의 경로를 넣고 실행하여 분석을 시작합니다. 
 
 ##### 비디오 분석은 videomain.py 파일을 실행하여 분석을 시작합니다. 파일을 실행하면 "주소를 입력하세요" 라는 문구가 나오고 그곳에 영상 경로를 입력하면 됩니다.
```
python audiomain.py
python videomain.py
```


### 🚀 spAIk-ai 실행 가이드 (Docker)
#### 이 프로젝트는 Docker를 사용하여 컨테이너 환경에서 쉽게 실행할 수 있습니다. 아래 가이드에 따라 프로젝트를 빌드하고 실행하세요.

#### 1. 사전 준비
##### Docker Desktop 설치 및 실행: Docker 컨테이너를 관리하기 위한 필수 도구입니다.

#### 2. 실행 명령어
##### 아래 순서대로 명령어를 입력하여 프로젝트를 실행합니다.
```
Bash

# 1. GitHub에서 프로젝트 클론
git clone https://github.com/spAIk-GS/spAIk_ai.git

# 2. 클론된 폴더로 이동
cd spAIk_ai

# 3. Docker 이미지 빌드
# 이미지 이름은 소문자로 지정해야 합니다.
docker build -t spaik_ai .

# 4. Docker 컨테이너 실행
# 애플리케이션은 5000번 포트에서 실행됩니다.
docker run -p 5000:5000 spaik_ai
```
#### 3. 애플리케이션 접속
##### 위 명령어를 실행하면 애플리케이션이 백그라운드에서 실행됩니다. 웹 브라우저를 열고 다음 주소로 접속하여 애플리케이션을 확인하세요.

- http://localhost:5000
