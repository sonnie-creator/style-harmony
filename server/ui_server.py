# project/server/ui_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional
from datetime import datetime 
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import traceback
from app_globals.objects import validator
from app_globals.utils import remove_background, image_to_base64, create_outfit_collage_v3
from PIL import Image

# ==============
app = FastAPI(title="Fashion Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic 모델 정의 ====================
class RecommendationRequest(BaseModel):
    prompt: str
    gender: str
    age: Optional[str] = None
    personalColor: Optional[str] = None
    season: Optional[str] = None
    user_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_id: Optional[str] = None
    outfit_id: str
    feedback_type: str
    outfit_items: dict[str, Any]
    prompt: str
    metadata: Optional[dict] = None

# ==================== 전역 상태 ====================
mcp_client_session = None
mcp_exit_stack = None
server_status = {
    "mcp_connected": False,
    "mcp_tools": [],
    "startup_logs": [],
    "last_error": None
}

BASE_DIR = os.path.abspath(".")
IMAGE_DIR = os.path.join(BASE_DIR, "./data_sources")

def log_status(message: str, level: str = "info"):
    """상태 로그 기록"""
    log_entry = {
        "timestamp": time.time(),
        "level": level,
        "message": message
    }
    server_status["startup_logs"].append(log_entry)
    print(f"[{level.upper()}] {message}", file=sys.stderr, flush=True)

# ==================== MCP 클라이언트 ====================
async def connect_mcp_client():
    """MCP 서버 연결"""
    global mcp_client_session, mcp_exit_stack, server_status

    try:
        log_status("🔧 MCP 서버 연결 시작...")
        
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.join(os.path.dirname(__file__), "mcp_server.py")],
            env=None
        )
        
        log_status("📡 Stdio transport 초기화 중...")
        mcp_exit_stack = AsyncExitStack()
        stdio_transport = await mcp_exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        
        log_status("🔗 ClientSession 생성 중...")
        read, write = stdio_transport
        mcp_client_session = await mcp_exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        
        log_status("🚀 MCP 서버 초기화 중...")
        await mcp_client_session.initialize()
        
        tools_list = await mcp_client_session.list_tools()
        server_status["mcp_tools"] = [
            {"name": tool.name, "description": tool.description}
            for tool in tools_list.tools
        ]
        
        server_status["mcp_connected"] = True
        server_status["last_error"] = None
        
        log_status(f"✅ MCP 클라이언트 연결 완료 ({len(server_status['mcp_tools'])}개 도구 사용 가능)", "success")
        
    except Exception as e:
        error_msg = f"MCP 클라이언트 연결 실패: {str(e)}"
        log_status(f"⚠️ {error_msg}", "warning")
        server_status["mcp_connected"] = False
        server_status["last_error"] = error_msg
        mcp_client_session = None

async def call_mcp_tool(tool_name: str, arguments: dict):
    """MCP Tool 호출"""
    if not mcp_client_session:
        raise Exception("MCP 클라이언트가 연결되지 않았습니다")
    
    log_status(f"🔧 MCP Tool 호출: {tool_name}")
    result = await mcp_client_session.call_tool(tool_name, arguments)
    
    if not result.content or len(result.content) == 0:
        log_status(f"⚠️ MCP Tool '{tool_name}' 응답 없음", "warning")
        return None
    
    text = next(c.text for c in result.content if getattr(c, "type", None) == "text")
    
    if not text or text.strip() == "":
        log_status(f"⚠️ MCP Tool '{tool_name}' 빈 응답 반환", "warning")
        return None
    
    try:
        data = json.loads(text)
        
        if "error" in data:
            log_status(f"❌ MCP Tool '{tool_name}' 에러: {data['error']}", "error")
            return None
        
        log_status(f"✅ MCP Tool '{tool_name}' 응답 성공", "success")
        return data
    except json.JSONDecodeError as e:
        log_status(f"❌ JSON 파싱 실패: {str(e)}", "error")
        raise Exception(f"MCP Tool 응답이 올바른 JSON이 아닙니다: {e}")

# ==================== FastAPI 이벤트 ====================
@app.on_event("startup")
async def startup_event():
    log_status("🚀 FastAPI 서버 시작...")
    log_status("📦 데이터 디렉토리 확인 중...")
    
    await connect_mcp_client()
    
    if server_status["mcp_connected"]:
        log_status("✅ 서버 준비 완료 (MCP 모드)", "success")
    else:
        log_status("✅ 서버 준비 완료 (Direct 모드)", "success")

@app.on_event("shutdown")
async def shutdown_event():
    global mcp_exit_stack
    log_status("👋 서버 종료 중...")
    if mcp_exit_stack:
        await mcp_exit_stack.aclose()
    log_status("✅ 서버 종료 완료", "success")

# ==================== API Endpoints ====================
@app.get("/")
async def root():
    return {
        "service": "Fashion Recommendation API",
        "status": "running",
        "mcp_connected": server_status["mcp_connected"],
        "available_tools": len(server_status["mcp_tools"])
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mcp_status": "connected" if server_status["mcp_connected"] else "disconnected",
        "mcp_tools": server_status["mcp_tools"],
        "startup_logs": server_status["startup_logs"][-10:],
        "last_error": server_status["last_error"]
    }

@app.get("/status")
async def get_status():
    """서버 상태 상세 정보"""
    return {
        "mcp_connected": server_status["mcp_connected"],
        "mcp_tools": server_status["mcp_tools"],
        "startup_logs": server_status["startup_logs"],
        "last_error": server_status["last_error"],
        "image_dir_exists": os.path.exists(IMAGE_DIR),
        "total_logs": len(server_status["startup_logs"])
    }
def json_safe(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

@app.post("/feedback")
async def save_feedback(feedback: FeedbackRequest):
    """사용자 피드백 저장 (user_id 없으면 저장 안함)"""
    try:
        log_status("📥 피드백 요청 수신", "info")

        # 1. 익명 사용자 처리
        if not feedback.user_id or feedback.user_id.strip() == "":
            log_status("💡 익명 사용자 - 피드백 저장 스킵", "info")
            return {
                "status": "skipped",
                "message": "익명 사용자는 피드백이 저장되지 않습니다",
                "is_anonymous": True
            }

        user_id = feedback.user_id.strip()

        # 2. feedback_type 검증
        if feedback.feedback_type not in ("like", "dislike"):
            log_status(f"❌ 잘못된 feedback_type: {feedback.feedback_type}", "error")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feedback_type: {feedback.feedback_type}"
            )

        # 3. 저장 데이터 구성
        feedback_data = {
            "user_id": user_id,
            "outfit_id": feedback.outfit_id,
            "feedback_type": feedback.feedback_type,
            "outfit_items": json_safe(feedback.outfit_items),
            "prompt": feedback.prompt,
            "timestamp": datetime.now().isoformat(),
            "metadata": json_safe(feedback.metadata or {})
        }

        # 4. 경로 준비
        user_json_dir = os.path.join(BASE_DIR, "data_sources", "user_json")
        os.makedirs(user_json_dir, exist_ok=True)
        user_json_path = os.path.join(user_json_dir, f"{user_id}.json")

        # 5. 기본 구조
        default_structure = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "preferences": {
                "liked_items": [],
                "disliked_items": [],
                "preferred_styles": [],
                "disliked_colors": []
            },
            "history": []
        }

        # 6. 기존 파일 로드
        if os.path.exists(user_json_path):
            try:
                with open(user_json_path, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
            except json.JSONDecodeError as e:
                log_status(
                    f"❌ JSON 파싱 실패 ({user_json_path}): line {e.lineno}, col {e.colno}",
                    "error"
                )
                user_data = default_structure.copy()
            except Exception as e:
                log_status(
                    f"❌ 사용자 파일 로드 실패 ({user_json_path}): {str(e)}",
                    "error"
                )
                user_data = default_structure.copy()
        else:
            user_data = default_structure.copy()

        # 7. 구조 보정
        user_data.setdefault("preferences", default_structure["preferences"].copy())
        user_data["preferences"].setdefault("liked_items", [])
        user_data["preferences"].setdefault("disliked_items", [])
        user_data["preferences"].setdefault("preferred_styles", [])
        user_data["preferences"].setdefault("disliked_colors", [])
        user_data.setdefault("history", [])

        # 8. 피드백 반영
        if feedback.feedback_type == "like":
            user_data["preferences"]["liked_items"].append(feedback_data)
        else:
            user_data["preferences"]["disliked_items"].append(feedback_data)

        user_data["history"].append(feedback_data)

        # 9. 파일 저장
        try:
            with open(user_json_path, "w", encoding="utf-8") as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log_status(
                f"❌ 사용자 파일 저장 실패 ({user_json_path}): {str(e)}",
                "error"
            )
            raise HTTPException(status_code=500, detail="파일 저장 실패")

        log_status(
            f"✅ 피드백 저장 완료 | user_id={user_id} | type={feedback.feedback_type}",
            "success"
        )

        return {
            "status": "success",
            "message": "피드백이 저장되었습니다",
            "user_id": user_id,
            "is_anonymous": False,
            "feedback_type": feedback.feedback_type,
            "stats": {
                "total_likes": len(user_data["preferences"]["liked_items"]),
                "total_dislikes": len(user_data["preferences"]["disliked_items"]),
                "total_history": len(user_data["history"])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        log_status("💥 피드백 저장 중 치명적 오류 발생", "error")
        log_status(str(e), "error")
        raise HTTPException(status_code=500, detail="서버 내부 오류")

        
@app.post("/recommend")
async def recommend_outfit(request: RecommendationRequest):
    """아웃핏 추천 API"""
    if server_status["mcp_connected"] and mcp_client_session:
        log_status(f"🎨 MCP 모드로 추천 시작: {request.prompt[:50]}...")
        
        result = await call_mcp_tool(
            "full_recommendation_pipeline",
            {
                "prompt": request.prompt or "",
                "gender": request.gender or "",
                "age": str(request.age) if request.age is not None else "",
                "personal_color": request.personalColor or "",
                "season": request.season or "",
                "user_id": str(request.user_id) if request.user_id is not None else ""
            }
        )

        if result:
            outfits = result.get("outfits", [])
            processed_outfits = []
            for outfit in outfits:
                processed_outfits.append({
                    'outfit_id': outfit.get('outfit_id'),
                    'items': outfit.get('items', {}),
                    'categories': list(outfit.get('items', {}).keys()),
                    'collage': outfit.get('collage'),
                    'scores': {
                        'validation': outfit.get('validation', {}).get('validation_score', 0),
                        'validation_details': outfit.get('validation', {}).get('details', {}),
                        'accepted': outfit.get('validation', {}).get('accepted', False)
                    }
                })
            
            response = {
                'prompt': request.prompt,
                'outfits': processed_outfits,
                'mode': 'mcp',
                'metadata': {
                    'total_outfits': len(processed_outfits),
                    'gender': request.gender or "",
                    'age': request.age or "",
                    'personalColor': request.personalColor or "",
                    'user_id': request.user_id or "",
                    'personalized': request.user_id is not None
                }
            }
            log_status(f"✅ MCP 추천 완료: {len(processed_outfits)}개 코디", "success")
            return response
        


async def process_outfits(outfits, request):
    """아웃핏 처리 공통 로직"""
    processed_outfits = []
    
    for outfit in outfits:
        processed_items = {}
        pil_items = {}

        try:
            for cat, item_data in outfit['items'].items():
                img_path = os.path.join(IMAGE_DIR, item_data['image_path'])
                if not os.path.exists(img_path):
                    continue
                
                original_img = Image.open(img_path).convert("RGB")
                transparent_img = remove_background(original_img)
                
                processed_items[cat] = {
                    'id': item_data['id'],
                    'name': item_data['name'],
                    'style': item_data['style'],
                    'color': item_data['color'],
                    'image': image_to_base64(original_img),
                    'image_transparent': image_to_base64(transparent_img),
                }
                pil_items[cat] = transparent_img

            try:
                collage = create_outfit_collage_v3(pil_items)
                collage_base64 = f"data:image/png;base64,{image_to_base64(collage)}"
            except Exception:
                collage_base64 = None

            outfit_item_ids = [item['id'] for item in outfit['items'].values()]
            try:
                val_score, accepted, val_details = validator.evaluate(
                    outfit_items=outfit_item_ids,
                    metadata={
                        "prompt": request.prompt or "",
                        "gender": request.gender or "",
                        "age": str(request.age) if request.age is not None else "",
                        "personal_color": request.personalColor or "",
                        "season": request.season or ""
                    },
                    user_id=str(request.user_id) if request.user_id is not None else None
                )
            except Exception:
                val_score, accepted, val_details = 0.0, False, {}

            processed_outfits.append({
                'outfit_id': outfit.get('outfit_id', None),
                'items': processed_items,
                'categories': outfit.get('categories', list(outfit['items'].keys())),
                'collage': collage_base64,
                'scores': {
                    **outfit.get('scores', {}),
                    'validation': val_score,
                    'validation_details': val_details,
                    'accepted': accepted
                }
            })
        except Exception:
            continue

    return processed_outfits



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)