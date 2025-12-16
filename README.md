
# Prompt-Based Fashion Outfit Recommendation System
<img width="1887" height="895" alt="image" src="https://github.com/user-attachments/assets/bbec83fd-66bb-4f42-9d69-09600b1df7a6" />

This project is a **prompt-driven fashion outfit recommendation system** built with an **MCP Tool Server** and a **FastAPI server**.
It provides an end-to-end pipeline including outfit recommendation, validation, and collage generation.

---

## Features

* **Prompt Analysis & Translation** (`analyze_prompt`)
* **Style-Based Outfit Recommendation** (`recommend_outfits`)
* **Outfit Validation** (`validate_outfit`)
* **Outfit Collage Generation** (`create_outfit_collage`)
* **Full Recommendation Pipeline** (`full_recommendation_pipeline`)
* **MCP Mode Support**

> Core functionalities are modularized into Tool units to ensure **reusability, maintainability, and extensibility**.

---

## Requirements

* Python 3.10 or higher
* Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Run the MCP Tool Server

If the MCP server is running, the FastAPI server will automatically operate in **MCP mode**.

```bash
python mcp_tool_server.py
```

#### Available Tools

| Tool Name                      | Description                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------- |
| `analyze_prompt`               | Detects input language and translates the prompt                              |
| `recommend_outfits`            | Recommends 3 outfits based on style prompt                                    |
| `validate_outfit`              | Validates whether an outfit combination is appropriate                        |
| `create_outfit_collage`        | Generates a collage image from outfit items                                   |
| `full_recommendation_pipeline` | Executes the full pipeline (analysis → recommendation → validation → collage) |

---

### 2. Run the FastAPI Server

```bash
uvicorn project.server.ui_server:app --host 0.0.0.0 --port 8002
```

* **MCP server connected** → MCP mode
* **MCP server not connected** → Direct mode (fallback)

---

### 3. API Request Example

```bash
curl -X POST "http://127.0.0.1:8002/recommend" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Summer casual outfit",
  "gender": "Women"
}'
```

---

## System Architecture

### MCP Tool Server

* Core logic is separated into independent Tools
* Each Tool follows a **single responsibility principle**
* Tools can be reused, combined, or extended independently

```
Client
  ↓ HTTP
FastAPI Server
  └─ MCP connected → MCP Tool Server
                       ├─ analyze_prompt
                       ├─ recommend_outfits
                       ├─ validate_outfit
                       └─ create_outfit_collage
```

---

### FastAPI Server

* Supports both **MCP mode** and **Direct mode**
* Automatically detects MCP availability at runtime
* Aggregates recommendation results, validation output, and collage images into a single API response

---

## Outfit Processing Pipeline 

1. **Prompt Analysis & Translation**
2. **Prompt Enrichment**

   * Age
   * Gender
   * Personal color
   * Season
3. **Outfit Recommendation**
    PPO-Based Recommendation Model (MCP Tool)

The core outfit recommendation logic inside the MCP Tool Server is powered by a PPO-trained reinforcement learning model.

Training Overview

The model was trained using Proximal Policy Optimization (PPO) in a custom Gymnasium environment for sequential outfit composition.

Each episode samples a fashion prompt and the agent selects items step by step to form a complete outfit.

Training prompts are derived from the neuralwork/fashion-style-instruct dataset.

Reward Design

The base environment reward reflects prompt–item semantic similarity and outfit coherence.

A ValidationAgent evaluates the final outfit and injects a validation score into the terminal reward:

final_reward =
  (1 - validation_weight) * env_reward
  + validation_weight * validation_score


During training, user personalization is disabled to avoid user-specific bias.
Personalization is applied only at inference time via MCP Tools.

Integration into MCP

The trained PPO model is loaded inside the MCP Tool Server.

MCP tools such as recommend_outfits and full_recommendation_pipeline internally invoke the PPO policy to generate outfit candidates.

The MCP layer handles prompt analysis, validation, and collage generation around the PPO-based core model. Models are available at: [Sonnie108/ppo-fashion-harmony](https://huggingface.co/Sonnie108/ppo-fashion-harmony)

   * MCP Tool-based pipeline or Direct PPO-based model
4. **Outfit Validation**
5. **Collage Generation**

   * Background removal
   * Transparent image composition
Item images are downloaded on demand via Hugging Face Hub: [Sonnie108/style-harmony](https://huggingface.co/datasets/Sonnie108/style-harmony)
Images are fetched in real time and cached locally
Transparent PNGs are composed into a single outfit collage layout

6. **Final Response**

   * JSON output including Base64-encoded collage images

---

## Example Recommendation Output

**Input**

```json
{
  "prompt": "Minimal summer outfit for casual office wear",
  "gender": "Women",
  "season": "Summer"
}
```

**Output (simplified)**

```json
{
  "outfits": [
    {
      "outfit_id": "outfit_001",
      "items": {
        "top": { "id": "top_123", "name": "Linen Blouse" },
        "bottom": { "id": "bottom_456", "name": "Wide Slacks" },
        "shoes": { "id": "shoes_789", "name": "Loafers" }
      },
      "validation": {
        "validation_score": 0.87,
        "accepted": true
      },
      "collage": "data:image/png;base64,iVBORw0KGgoAAA..."
    }
  ],
  "original_prompt": "Minimal summer outfit for casual office wear",
  "translated_prompt": "Minimal summer outfit for casual office wear"
}
```

The response includes:

* Recommended outfit items
* Validation score and acceptance result
* A Base64-encoded collage image ready for frontend rendering

---

## Summary

* MCP Tool Server handles **all heavy logic**
* FastAPI acts as a **unified API gateway**
* Tool-based design ensures:

  * Clean separation of concerns
  * Easier debugging and testing
  * Scalable future extensions

---

## License

MIT License

```

