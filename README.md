# 🦺 작업장 안전 관리 CCTV 시스템

> **현대오토에버 모빌리티 SW 스쿨 스마트 팩토리 과정 · 4조**  
> 머신비전 기반 실시간 작업자 안전 모니터링 시스템

<br>

## 📹 시연 영상


https://drive.google.com/file/d/16ircE3qgpiMDMTtXbDWnLL1Osywpa-yU/view?usp=drive_link

<br>

## 📌 프로젝트 개요

건설·제조 현장의 안전사고를 예방하기 위해 CCTV 영상을 실시간으로 분석하는 AI 안전 모니터링 시스템입니다.

작업자가 많고 위험 요소가 혼재하는 현장에서 **관리자가 모든 상황을 직접 감시하기 어렵다**는 문제를 해결하기 위해,  
AI가 실시간으로 작업자를 인식하고 안전 보호구 착용 여부와 낙상 위험을 자동으로 탐지합니다.

<br>

## ✨ 주요 기능

| 기능 | 설명 | 기술 |
|------|------|------|
| **얼굴 인식** | 등록된 작업자 신원 자동 식별 | face_recognition (dlib HOG) |
| **PPE 감지** | 헬멧·장갑·안전조끼 착용 여부 판단 | YOLO26s (Custom) |
| **포즈 추정** | 17개 키포인트 기반 자세 분석 | YOLO26s-Pose |
| **낙상 감지** | 다중 조건 Rule-based 낙상 판단 | YOLO Pose + Rule-based |
| **실시간 스트리밍** | MJPEG 스트리밍 + REST API | Flask |

<br>

## 📊 성능 지표

### 얼굴 인식 &nbsp;|&nbsp; 전체 정확도 **97.38%**

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 재현 (Jaehyun) | 1.00 | 0.93 | 0.97 | 45 |
| 지원 (Jiwon) | 1.00 | 1.00 | 1.00 | 50 |
| 명재 (Myeongjae) | 0.98 | 0.98 | 0.98 | 42 |
| 외부인 (Unknown) | 0.93 | 0.98 | 0.95 | 54 |
| **전체** | **0.98** | **0.97** | **0.97** | **191** |

### PPE 감지 &nbsp;|&nbsp; 모델 비교 (train : val : test = 7 : 2 : 1)

| 모델 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| YOLO26n | 0.884 | 0.810 | 0.868 | 0.710 |
| **YOLO26s** ✅ | **0.899** | **0.816** | **0.876** | **0.719** |
| YOLO26m | 0.890 | 0.809 | 0.874 | 0.690 |

> YOLO26s — Precision과 Recall을 균형 있게 유지하면서 가장 높은 mAP 기록으로 최종 채택

### 낙상 감지

| 지표 | 값 |
|------|----|
| 전체 정확도 (Accuracy) | 71.4% |
| Precision | 76.9% |
| Recall | 76.9% |
| F1-Score | 0.77 |

<br>

## 🗂️ 프로젝트 구조

```
SF4-Machine-Vision-Project/
│
├── config.py              # 모든 설정값 (모델 경로, 임계값, 표시 옵션)
├── processor.py           # 핵심 프레임 처리 로직 (app.py · run.py 공유)
├── app.py                 # Flask 웹서버 — MJPEG 스트리밍 + REST API
├── run.py                 # 독립 실행 — 로컬 창 표시 / 영상 저장
├── requirements.txt
│
├── core/
│   ├── utils.py           # 기하 유틸 (IoU, distance, bbox 등)
│   ├── ppe.py             # PPE 감지 (헬멧·장갑·조끼 매칭)
│   ├── face.py            # 얼굴 인식
│   ├── fall.py            # 낙상 감지 (히스토리 기반 Rule-based)
│   ├── tracker.py         # 사람 추적 (IoU 매칭)
│   ├── draw.py            # 시각화 (박스·키포인트·텍스트)
│   └── logger.py          # 로그 관리 (thread-safe)
│
├── best_v04_03.pt         # PPE 감지 커스텀 모델
├── yolo26s-pose.pt        # Pose 추정 모델
└── trained_faces_0320.pkl # 등록 작업자 얼굴 인식 DB
```

<br>

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

> CUDA 환경에서 GPU 가속을 사용하려면 PyTorch CUDA 버전을 별도로 설치하세요.  
> https://pytorch.org/get-started/locally/

<br>

## 🚀 실행

### 웹서버 모드 (Flask)

```bash
python app.py
```

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/video_feed` | GET | 감지 결과 스트림 (MJPEG) |
| `/raw_feed` | GET | 원본 카메라 스트림 |
| `/api/logs` | GET | 감지 로그 (JSON) |
| `/api/reset_logs` | POST | 로그 초기화 |
| `/api/set_source` | POST | 입력 소스 변경 `{"source": "0"}` |

**접속 예시**
```
http://localhost:5000/video_feed?source=0           # 웹캠 0번
http://localhost:5000/video_feed?source=1           # 웹캠 1번
http://localhost:5000/video_feed?source=video.mp4  # 영상 파일
http://localhost:5000/video_feed?source=1&width=800&quality=55  # 경량 스트리밍
```

### 독립 실행 모드

```bash
python run.py
```

| 단축키 | 동작 |
|--------|------|
| `q` / `ESC` | 종료 |
| `Space` | 일시정지 / 재개 |

<br>

## 🔧 주요 설정 (`config.py`)

```python
# 입력 소스 (독립 실행)
DEFAULT_INPUT_SOURCE = 1        # 웹캠 인덱스 또는 "video.mp4"

# 모델 경로
POSE_MODEL_PATH = "yolo26s-pose.pt"
PPE_MODEL_PATH  = "best_v04_03.pt"
PKL_PATH        = "trained_faces_0320.pkl"

# 얼굴 인식
FACE_TOLERANCE       = 0.75    # 낮을수록 엄격
FACE_RECOG_INTERVAL  = 8       # 인식 주기 (프레임)

# 낙상 감지
FALL_CONFIRM_FRAMES  = 4       # 낙상 확정 프레임 수
FALL_RECOVER_FRAMES  = 12      # 정상 복구 프레임 수
```

<br>

## 🛠️ 기술 스택

- **언어**: Python 3.9+
- **딥러닝**: PyTorch, Ultralytics YOLO
- **컴퓨터 비전**: OpenCV, face_recognition (dlib)
- **웹서버**: Flask, Flask-CORS
- **GPU 가속**: CUDA — YOLO 추론 / CPU HOG — 얼굴 인식
