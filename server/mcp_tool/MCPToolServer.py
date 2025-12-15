"""
MCP Tool Server - Fashion Recommendation System
모든 핵심 로직을 MCP Tool로 제공
"""
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.types as types
import json
import os
from PIL import Image
from huggingface_hub import hf_hub_download
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

class MCPToolServer:
    def __init__(self, name="fashion-reco-mcp"):
        self.server = Server(name)
        self._register_handlers()
    
    def _register_handlers(self):
        """핸들러 등록"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """사용 가능한 Tool 목록 반환"""
            return [
                types.Tool(
                    name="analyze_prompt",
                    description="프롬프트를 분석하고 번역. 언어 감지 및 영어 변환",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "사용자 입력 프롬프트"},
                        },
                        "required": ["prompt"]
                    }
                ),
                types.Tool(
                    name="recommend_outfits",
                    description="스타일 프롬프트 기반 아웃핏 추천 (3개)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "번역된 프롬프트"},
                            "gender": {"type": "string", "description": "성별 (Men/Women/All)"},
                            "age": {"type": "string", "description": "나이대 (선택)"},
                            "personal_color": {"type": "string", "description": "퍼스널 컬러 (선택)"},
                            "season": {"type": "string", "description": "계절 (선택)"}
                        },
                        "required": ["prompt", "gender"]
                    }
                ),
                types.Tool(
                    name="validate_outfit",
                    description="아웃핏 조합이 어울리는지 검증",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "item_ids": {
                                "type": "array", 
                                "items": {"type": "string"}, 
                                "description": "아이템 ID 리스트"
                            },
                            "user_id": {"type": "string", "description": "사용자 ID (선택)"}
                        },
                        "required": ["item_ids"]
                    }
                ),
                types.Tool(
                    name="create_outfit_collage",
                    description="아웃핏 아이템들을 콜라주 이미지로 생성",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "items": {"type": "object", "description": "카테고리별 아이템 정보"},
                        },
                        "required": ["items"]
                    }
                ),
                types.Tool(
                    name="full_recommendation_pipeline",
                    description="전체 추천 파이프라인 (분석→추천→검증→콜라주 생성)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "사용자 입력"},
                            "gender": {"type": "string", "description": "성별"},
                            "age": {"type": "string", "description": "나이대 (선택)"},
                            "personal_color": {"type": "string", "description": "퍼스널 컬러 (선택)"},
                            "season": {"type": "string", "description": "계절 (선택)"},
                            "user_id": {"type": "string", "description": "사용자 ID (선택)"}
                        },
                        "required": ["prompt"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, 
            arguments: dict | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Tool 실행"""
            
            if arguments is None:
                arguments = {}
            
            try:
                if name == "analyze_prompt":
                    result = await self._analyze_prompt(arguments)
                
                elif name == "recommend_outfits":
                    result = await self._recommend_outfits(arguments)
                
                elif name == "validate_outfit":
                    result = await self._validate_outfit(arguments)
                
                elif name == "create_outfit_collage":
                    result = await self._create_collage(arguments)
                
                elif name == "full_recommendation_pipeline":
                    result = await self._full_pipeline(arguments)
                
                else:
                    result = [types.TextContent(
                        type="text", 
                        text=json.dumps({"error": f"Unknown tool: {name}"})
                    )]
                
                return result
            
            except Exception as e:
                import traceback
                error_msg = f"Error in {name}: {str(e)}\n{traceback.format_exc()}"
                return [types.TextContent(type="text", text=json.dumps({"error": error_msg}))]
    
    # ==================== Tool 구현 메서드 ====================
    
    async def _analyze_prompt(self, args: dict) -> list[types.TextContent]:
        """프롬프트 분석 및 번역"""
        from app_globals.utils import detect_and_translate
        
        prompt = args.get("prompt", "")
        detected_lang, translated = detect_and_translate(prompt)
        
        result = {
            "original_language": detected_lang,
            "translated_prompt": translated
        }
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    
    async def _recommend_outfits(self, args: dict) -> list[types.TextContent]:
        """아웃핏 추천"""
        from app_globals.objects import recommender
        
        kwargs = {
            "prompt": args.get("prompt"),  # 필수는 아니면 None 가능
            "gender": args.get("gender"),
            "age": args.get("age"),
            "personal_color": args.get("personal_color"),
            "season": args.get("season"),
            "num_outfits": 3
        }

        # 값이 없는 인자는 제거
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        outfits = recommender.recommend_outfits(**kwargs)

        return [types.TextContent(type="text", text=json.dumps(outfits, ensure_ascii=False))]
    
    async def _validate_outfit(self, args: dict) -> list[types.TextContent]:
        """아웃핏 검증"""
        from app_globals.objects import validator
        
        item_ids = args.get("item_ids", [])
        user_id = args.get("user_id")
        
        score, accepted, details = validator.evaluate(item_ids, {}, user_id)
        
        result = {
            "validation_score": score,
            "accepted": accepted,
            "details": details
        }
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    
    async def _create_collage(self, args: dict) -> list[types.TextContent]:
        """콜라주 이미지 생성 (HF Hub 연동)"""
        from app_globals.utils import create_outfit_collage_v3, remove_background, image_to_base64
        from huggingface_hub import hf_hub_download
        import os
        from PIL import Image
        import sys
        
        items = args.get("items", {})
        pil_items = {}

        repo_id = "Sonnie108/style-harmony"
        repo_type = "dataset"
        local_root = "./server/data_sources/images"
        os.makedirs(local_root, exist_ok=True)

        # HF 레포에 있는 폴더 리스트
        hf_folders = ["images_part1", "images_part2", "images_part3"]

        for cat, item_data in items.items():
            img_file = os.path.basename(item_data.get("image_path", ""))
            if not img_file:
                print(f"[Warning] {cat} 아이템에 image_path 없음", file=sys.stderr, flush=True)
                continue
            
            # 로컬 파일 경로 (루트에 직접 저장)
            local_path = os.path.join(local_root, img_file)

            # 로컬에 없으면 HF Hub에서 다운로드 시도
            if not os.path.exists(local_path):
                downloaded = False
                for folder in hf_folders:
                    hf_file_path = f"{folder}/{img_file}"
                    try:
                        # ⭐ hf_hub_download는 원본 경로 구조를 유지하므로
                        # local_dir_use_symlinks=False로 실제 파일 복사
                        downloaded_path = hf_hub_download(
                            repo_id=repo_id,
                            repo_type=repo_type,
                            filename=hf_file_path,
                            local_dir=local_root,
                            local_dir_use_symlinks=False
                        )
                        
                        # ⭐ 다운로드된 파일을 루트로 이동
                        # downloaded_path는 ./server/data_sources/images/images_part1/xxx.jpg
                        if os.path.exists(downloaded_path) and downloaded_path != local_path:
                            import shutil
                            shutil.move(downloaded_path, local_path)
                            print(f"[Success] {img_file} 다운로드 완료", file=sys.stderr, flush=True)
                        
                        downloaded = True
                        break
                        
                    except Exception as e:
                        print(f"[Debug] {folder}에서 {img_file} 다운로드 시도 실패: {e}", file=sys.stderr, flush=True)
                        continue
                
                if not downloaded:
                    print(f"[Warning] {img_file} 다운로드 실패 (모든 폴더 시도 완료)", file=sys.stderr, flush=True)
                    continue

            # 이미지 열기 및 배경 제거
            try:
                if os.path.exists(local_path):
                    img = Image.open(local_path).convert("RGBA")
                    pil_items[cat] = remove_background(img)
                else:
                    print(f"[Warning] {local_path} 파일 없음", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[Error] {img_file} 이미지 처리 실패: {e}", file=sys.stderr, flush=True)
                continue

        # 콜라주 생성
        if not pil_items:
            print(f"[Warning] 콜라주 생성 가능한 이미지 없음", file=sys.stderr, flush=True)
            return [types.TextContent(type="text", text=json.dumps({"collage_base64": ""}))]
        
        try:
            collage = create_outfit_collage_v3(pil_items)
            collage_b64 = image_to_base64(collage)
            result = {"collage_base64": f"data:image/png;base64,{collage_b64}"}
            print(f"[Success] 콜라주 생성 완료, base64 길이: {len(collage_b64)}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[Error] 콜라주 생성 실패: {e}", file=sys.stderr, flush=True)
            result = {"collage_base64": ""}

        return [types.TextContent(type="text", text=json.dumps(result))]

    async def _full_pipeline(self, args: dict) -> list[types.TextContent]:
        """전체 파이프라인 실행 (진행 상태 및 처리된 프롬프트/결과 포함)"""
        # 1. 프롬프트 분석
        prompt = args.get("prompt", "")

        analyze_result = await self._analyze_prompt({"prompt": prompt})
        analysis = json.loads(analyze_result[0].text)

        # 2. 프롬프트 강화
        enriched_prompt = analysis["translated_prompt"]
        step_info = {"translated_prompt": enriched_prompt}
        if args.get("age"):
            enriched_prompt += f" for {args['age']} year old"
            step_info["age_info"] = args["age"]
        if args.get("gender") and args["gender"] != "All":
            enriched_prompt += f" {args['gender'].lower()}."
            step_info["gender_info"] = args["gender"]
        if args.get("personal_color"):
            enriched_prompt += f" Personal color is {args['personal_color']}."
            step_info["personal_color"] = args["personal_color"]
        if args.get("season"):
            enriched_prompt += f" Suitable for {args['season']} season."
            step_info["season"] = args["season"]

        # 3. 추천 생성
        recommend_kwargs = {k: v for k, v in {
            "prompt": enriched_prompt,
            "gender": args.get("gender"),
            "age": args.get("age"),
            "personal_color": args.get("personal_color"),
            "season": args.get("season")
        }.items() if v is not None}


        recommend_result = await self._recommend_outfits(recommend_kwargs)
        outfits = json.loads(recommend_result[0].text)

        # 4. 각 아웃핏 검증 및 콜라주 생성
        processed_outfits = []
        for i, outfit in enumerate(outfits):
            progress_base = 0.3 + 0.6 * (i / len(outfits))

            # 검증
            validate_result = await self._validate_outfit({
                "item_ids": [item["id"] for item in outfit["items"].values()],
                "user_id": args.get("user_id")
            })
            validation = json.loads(validate_result[0].text)

            # 콜라주 생성
            
            collage_result = await self._create_collage({"items": outfit["items"]})
            collage = json.loads(collage_result[0].text)
            collage_base64 = collage.get("collage_base64", "")
            
            processed_outfits.append({
                "outfit_id": outfit["outfit_id"],
                "items": outfit["items"],
                "validation": validation,
                "collage": collage["collage_base64"]
            })

        final_result = {
            "outfits": processed_outfits,
            "original_prompt": prompt,
            "translated_prompt": analysis["translated_prompt"]
        }

        return [types.TextContent(type="text", text=json.dumps(final_result, ensure_ascii=False))]

    
    async def run(self):
        """MCP 서버 실행 (stdio 모드)"""
        from mcp.server.stdio import stdio_server
        from mcp.types import ServerCapabilities, ToolsCapability
        from mcp.server.models import InitializationOptions
        import sys
        

        print("MCP Server initialized", file=sys.stderr, flush=True)
        
        async with stdio_server() as (read_stream, write_stream):
            # ServerCapabilities 직접 생성
            capabilities = ServerCapabilities(
                tools=ToolsCapability()
            )
            
            init_options = InitializationOptions(
                server_name="fashion-reco-mcp",
                server_version="1.0.0",
                capabilities=capabilities
            )
            
            await self.server.run(
                read_stream,
                write_stream,
                init_options
            )