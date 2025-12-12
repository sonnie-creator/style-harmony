generate_recommendation_tool = {
    "name": "recommend_outfit",
    "description": "사용자 선호 기반 추천",
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "gender": {"type": "string"}
        },
        "required": ["prompt"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "item_ids": {"type": "array", "items": {"type": "string"}},
            "links": {"type": "array", "items": {"type": "string"}}
        }
    }
}
