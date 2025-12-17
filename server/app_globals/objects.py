# app_globals/objects.py
import pickle
import os
import torch
from env.env_outfit_train import OutfitBatchRecommender, ValidationAgent, OutfitCompositionEnv, EncoderWrapper, UserLogManager
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from typing import Optional, Dict, List

base_dir = "./data_sources"
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = EncoderWrapper(device=device)

# ============== Dataset 로드 ==============
print("🔄 Loading dataset...")
with open("./data_sources/fashion_products.pkl", "rb") as f:
    dataset = pickle.load(f)

if isinstance(dataset, list):
    dataset = {item["product_code"]: item for item in dataset}

dataset = {str(k): v for k, v in dataset.items()}
print(f"✅ Dataset loaded: {len(dataset)} items")

# ============== PPO 모델 로드 ==============
print("🔄 Loading PPO model...")

def make_env():
    return OutfitCompositionEnv(
        "./data_sources/fashion_products.pkl", 
        encoder=encoder, 
        top_k=40
    )

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load("./model/vec_normalize.pkl", vec_env)
vec_env.training = False
vec_env.norm_reward = False

# PPO 모델 로드
model = PPO.load("./model/ppo_model", device=device)
print("✅ PPO model loaded")

# ============== User Log Manager (✅ 수정됨) ==============
print("🔄 Initializing User Log Manager...")
log_manager = UserLogManager(base_dir="./data_sources/user_json/")
print("✅ User Log Manager initialized (실시간 피드백 반영 모드)")

# ============== Validator (✅ Manager 객체 전달) ==============
print("🔄 Initializing Validator...")
validator = ValidationAgent(
    dataset=dataset,
    user_log_manager=log_manager,  # ✅ Manager 객체 전달 (user_logs 대신)
    min_score=0.5
)
print("✅ Validator initialized with real-time feedback support")

# ============== Recommender ==============
print("🔄 Initializing Recommender...")
recommender = OutfitBatchRecommender(
    env=vec_env.envs[0],
    dataset=dataset,
    model=model,
    vec_env=vec_env
)
print("✅ Recommender initialized")

print("\n" + "="*60)
print("🎉 All components loaded successfully!")
print("="*60)
print(f"Device: {device}")
print(f"Dataset items: {len(dataset)}")
print(f"Validation threshold: 0.5")
print(f"Real-time feedback: ✅ Enabled")
print("="*60 + "\n")
