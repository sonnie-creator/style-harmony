# app_globals/objects.py
import pickle
import os
import torch
from env.env_outfit_train import OutfitBatchRecommender, ValidationAgent, OutfitCompositionEnv, EncoderWrapper
from user.user_logs import UserLogManager
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from typing import Optional, Dict, List

base_dir = "./data_sources"
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = EncoderWrapper(device=device)
# Dataset 로드
print("🔄 Loading dataset...")
with open("./data_sources/fashion_products.pkl", "rb") as f:
        dataset = pickle.load(f)
if isinstance(dataset, list):
        dataset = {item["id"]: item for item in dataset}
    
filtered_dataset = {k: v for k, v in dataset.items() 
                       if v.get("image_path") and os.path.exists(os.path.join(base_dir, v["image_path"]))}
print(f"📌 Valid images: {len(filtered_dataset)}/{len(dataset)}")
dataset = filtered_dataset
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

# PPO 모델 로드 (zip)
model = PPO.load("./model/ppo_model", device=device)

# 벡터 임베딩 로드

log_manager = UserLogManager(log_file="user_logs.json")
validator = ValidationAgent(dataset=dataset, user_logs=log_manager.logs, min_score=0.7)
recommender = OutfitBatchRecommender(env=vec_env.envs[0],
    dataset=dataset,
    model=model,
    vec_env=vec_env
)