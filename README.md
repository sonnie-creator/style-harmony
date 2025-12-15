좋습니다. 지금까지의 내용과 MCP 서버 구조, Direct 모드, 데이터 준비, 실행, API 예시, Tool 단위 설명까지 포함해서 **전체 README.md** 파일 형태로 정리하면 아래와 같습니다.

````markdown
# Fashion Recommendation System

사용자 프롬프트 기반 패션 아웃핏 추천 시스템입니다. MCP 서버와 FastAPI 서버를 활용하여 아웃핏 추천, 검증, 콜라주 생성 등 전체 파이프라인을 제공합니다.

---

## Features

- **프롬프트 분석 및 번역** (`analyze_prompt`)
- **스타일 기반 아웃핏 추천** (`recommend_outfits`)
- **아웃핏 검증** (`validate_outfit`)
- **아웃핏 콜라주 생성** (`create_outfit_collage`)
- **전체 추천 파이프라인** (`full_recommendation_pipeline`)
- **MCP 모드와 Direct 모드 지원**  
  - MCP 서버 실행 시 Tool 기반 모드  
  - MCP 서버 미실행 시 Direct 모드 (PPO 모델 사용)

> Tool 단위로 기능을 분리하여 모듈화, 재사용성, 유지보수 용이성을 확보했습니다.

---

## Requirements

- Python 3.10 이상
- 필요한 라이브러리 설치:
```bash
pip install -r requirements.txt
````

---

## 데이터 준비

`data_sources` 폴더에 필요한 데이터와 이미지를 배치합니다.

1. Hugging Face Hub에서 `dataset.pkl` 다운로드:

```python
from huggingface_hub import hf_hub_download

dataset_file = hf_hub_download(
    repo_id="Sonnie108/style-harmony",
    filename="dataset.pkl"
)
```

2. 이미지 파일은 `data_sources` 폴더 내 카테고리별 서브폴더에 배치

---

## 실행 방법

### 1. MCP 서버 실행 (권장)

MCP 서버를 먼저 실행하면 FastAPI 서버는 MCP 모드로 동작합니다.

```bash
python mcp_tool_server.py
```

#### Tool별 기능

| Tool 이름                        | 기능                            |
| ------------------------------ | ----------------------------- |
| `analyze_prompt`               | 입력 프롬프트 언어 감지 및 번역            |
| `recommend_outfits`            | 스타일 기반 아웃핏 3개 추천              |
| `validate_outfit`              | 아웃핏 조합 검증                     |
| `create_outfit_collage`        | 아웃핏 아이템 콜라주 생성                |
| `full_recommendation_pipeline` | 전체 추천 파이프라인 실행 (분석→추천→검증→콜라주) |

### 2. FastAPI 서버 실행

```bash
uvicorn project.server.ui_server:app --host 0.0.0.0 --port 8002
```

* MCP 서버 연결 시: MCP 모드
* MCP 서버 미연결 시: Direct 모드

### 3. API 호출 예시

```bash
curl -X POST "http://127.0.0.1:8002/recommend" \
-H "Content-Type: application/json" \
-d '{
    "prompt": "Summer casual outfit",
    "gender": "Women"
}'
```

---

## Direct 모드

* MCP 서버 없이도 FastAPI 서버 단독 실행 가능
* 내부적으로 PPO 모델 기반 Direct 추천 수행
* 프롬프트 번역, 강화, 추천, 검증, 콜라주 생성 포함

---

## 구조 설명

* **MCP Tool Server 구조**

  * 핵심 로직을 Tool 단위로 분리
  * Tool 단위 분리 이유:

    1. 단일 책임 원칙(SRP) 준수
    2. 재사용성 확보
    3. 유연한 조합 가능
    4. 에러 격리 및 디버깅 용이
    5. 확장성 확보
* **FastAPI 서버**

  * MCP 모드와 Direct 모드를 모두 지원
  * API 호출 시 자동으로 MCP 연결 여부 확인 후 모드 선택
  * 추천 결과 처리 및 콜라주 생성, 검증까지 통합

---

## Outfits 처리 과정 (요약)

1. **프롬프트 분석 및 번역**
2. **프롬프트 강화** (나이, 성별, 퍼스널 컬러, 계절 등)
3. **추천 생성** (MCP Tool / Direct PPO 모델)
4. **아웃핏 검증** (Validator)
5. **콜라주 생성** (Transparent 이미지 + 배치)
6. **결과 반환** (JSON, Base64 이미지 포함)

---

## License

MIT License

```

