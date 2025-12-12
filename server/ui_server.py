# project/server/ui_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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

class RecommendationRequest(BaseModel):
    prompt: str
    gender: str
    age: Optional[str] = None
    personalColor: Optional[str] = None
    season: Optional[str] = None
    user_id: Optional[str] = None

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
    print(f"[{level.upper()}] {message}", flush=True)

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
        
        # 사용 가능한 도구 목록 가져오기
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
    
    if os.path.exists(IMAGE_DIR):
        log_status(f"✅ 이미지 디렉토리 확인: {IMAGE_DIR}", "success")
    else:
        log_status(f"⚠️ 이미지 디렉토리 없음: {IMAGE_DIR}", "warning")
    
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
        "startup_logs": server_status["startup_logs"][-10:],  # 최근 10개 로그
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

def search_image_url(product_name: str):
    query = product_name.replace(" ", "+")
    return f"https://www.google.com/search?tbm=isch&q={query}"

@app.post("/recommend")
async def recommend_outfit(request: RecommendationRequest):
    """아웃핏 추천 API"""
    try:
        # MCP 모드 시도
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
                processed_outfits = await process_outfits(outfits, request)
                
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

        # Direct 모드 fallback
        log_status("🔄 Direct 모드로 전환...", "warning")
        return await recommend_direct(request)

    except Exception as e:
        log_status(f"❌ 추천 실패: {str(e)}", "error")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
                    'image_url': search_image_url(item_data['name'])
                }
                pil_items[cat] = transparent_img

            # Collage 생성
            try:
                collage = create_outfit_collage_v3(pil_items)
                collage_base64 = f"data:image/png;base64,{image_to_base64(collage)}"
            except Exception:
                collage_base64 = None

            # Validation
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

async def recommend_direct(request: RecommendationRequest):
    """Direct 모드 추천"""
    from app_globals.objects import recommender
    from app_globals.utils import detect_and_translate
    
    log_status("🎨 Direct 모드 추천 시작...")
    
    # 번역 및 프롬프트 강화
    detected_lang, translated_prompt = detect_and_translate(request.prompt)
    enriched_prompt = translated_prompt
    
    if request.age:
        enriched_prompt += f" for {request.age} year old"
    if request.gender and request.gender != "All":
        enriched_prompt += f" {request.gender.lower()}."
    if request.personalColor:
        enriched_prompt += f" Personal color is {request.personalColor}."
    if request.season:
        enriched_prompt += f" Suitable for {request.season} season."
    
    # PPO 추천 실행
    outfits = recommender.recommend_outfits(
        prompt=enriched_prompt,
        gender=request.gender,
        age=request.age,
        personal_color=request.personalColor,
        num_outfits=3
    )
    
    # 아웃핏 처리
    processed_outfits = await process_outfits(outfits, request)
    
    response = {
        'prompt': str(request.prompt),
        'translatedPrompt': str(translated_prompt) if detected_lang != 'en' else None,
        'enrichedPrompt': str(enriched_prompt),
        'detectedLanguage': str(detected_lang),
        'outfits': processed_outfits,
        'mode': 'direct',
        'metadata': {
            'total_outfits': len(processed_outfits),
            'gender': request.gender,
            'age': request.age,
            'personalColor': request.personalColor,
            'user_id': request.user_id,
            'personalized': request.user_id is not None
        }
    }
    
    log_status(f"✅ Direct 추천 완료: {len(processed_outfits)}개 코디", "success")
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)