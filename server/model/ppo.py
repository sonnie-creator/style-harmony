# train_ppo_fashion_tqdm.py - 개인화 학습 버전
import random
import gymnasium as gym 
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import json
from env.env_outfit_train import OutfitCompositionEnv, EncoderWrapper, ValidationAgent, UserLogManager  # ✅ 사용자 로그
from datasets import load_dataset
from stable_baselines3.common.vec_env import VecNormalize
import wandb
from stable_baselines3.common.logger import configure
import pickle

# ===================================
# WandB 초기화
# ===================================
wandb.init(
    project="outfit-recommendation-ppo-personalized",
    config={
        "learning_rate": 1e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.02,
        "clip_range": 0.1,
        "use_personalization": True,  # ✅ 개인화 활성화
        "validation_weight": 0.3  # ✅ Validation 가중치
    }
)

# ===================================
# WandB 콜백
# ===================================
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # 에피소드 정보 로깅
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info["r"])
            self.episode_lengths.append(ep_info["l"])
            
            wandb.log({
                "train/reward": ep_info["r"],
                "train/ep_length": ep_info["l"],
                "train/reward_mean_100": np.mean(self.episode_rewards[-100:]),
            })

        # PPO 내부 지표
        if hasattr(self.model.logger, "name_to_value"):
            metrics = self.model.logger.name_to_value
            wandb.log({
                "train/explained_variance": metrics.get("train/explained_variance", None),
                "train/approx_kl": metrics.get("train/approx_kl", None),
                "train/clip_fraction": metrics.get("train/clip_fraction", None),
                "train/entropy_loss": metrics.get("train/entropy_loss", None),
                "train/value_loss": metrics.get("train/value_loss", None),
                "train/policy_gradient_loss": metrics.get("train/policy_gradient_loss", None),
                "train/loss": metrics.get("train/loss", None),
                "train/learning_rate": metrics.get("train/learning_rate", None),
            })

        return True

# ===================================
# 프롬프트 데이터셋 로드
# ===================================
print("📥 Loading fashion dataset...")
dataset = load_dataset("neuralwork/fashion-style-instruct", split="train")

prompts = []
for sample in dataset:
    input_text = sample["input"].strip()
    context_text = sample["context"].strip()
    prompt = f"{input_text}, event: {context_text} 룩 추천"
    prompts.append(prompt)

print(f"✅ Loaded {len(prompts)} prompts")

# ===================================
# 사용자 로그 및 Validation 초기화
# ===================================
print("📥 Loading user logs and validation agent...")
log_manager = UserLogManager()

# 데이터셋 로드 (Validation용)
with open("./data_sources/fashion_products.pkl", "rb") as f:
    full_dataset = pickle.load(f)
if isinstance(full_dataset, list):
    full_dataset = {item["id"]: item for item in full_dataset}

validator = ValidationAgent(
    dataset=full_dataset,
    user_logs=log_manager.logs,
    min_score=0.7
)

print(f"✅ Loaded {len(log_manager.logs)} user profiles")

# ===================================
# Gymnasium 래퍼들
# ===================================
class FashionEnvWrapper(gym.Wrapper):
    """프롬프트 자동 선택 래퍼"""
    def __init__(self, env: OutfitCompositionEnv, prompts: list, gender: str = "All"):
        super().__init__(env)
        self.prompts = prompts
        self.gender = gender
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        prompt_text = random.choice(self.prompts)
        obs, info = self.env.reset(prompt_text=prompt_text, gender=self.gender, seed=seed)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info


class FixedActionSpaceWrapper(gym.Wrapper):
    """고정된 액션 공간 래퍼"""
    def __init__(self, env, max_actions=50):
        super().__init__(env)
        self.max_actions = max_actions
        self.action_space = gym.spaces.Discrete(max_actions)
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
    
    def step(self, action):
        base_env = self.env.unwrapped
        pool_size = len(base_env.candidate_pool_for_step)
        
        if pool_size == 0:
            obs = base_env._get_observation()
            return obs, -1.0, True, False, {"error": "no_candidates"}
        
        actual_action = action % pool_size
        obs, reward, done, truncated, info = self.env.step(actual_action)
        return obs, reward, done, truncated, info


class ExplorationWrapper(gym.Wrapper):
    """Epsilon-greedy 탐색 래퍼"""
    def __init__(self, env, initial_epsilon=0.5, min_epsilon=0.05, decay_steps=5000):
        super().__init__(env)
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = (initial_epsilon - min_epsilon) / decay_steps
        self.step_count = 0
    
    def step(self, action):
        base_env = self.env.unwrapped
        pool_size = len(base_env.candidate_pool_for_step)
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon and pool_size > 0:
            action = np.random.randint(0, pool_size)
        
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.step_count += 1
        self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate)
        
        return obs, reward, done, truncated, info


class ValidationRewardWrapper(gym.Wrapper):
    """Validation 기반 리워드 보정 (옵션으로 개인화 사용 여부 결정)"""
    def __init__(self, env, validator: ValidationAgent,
                 validation_weight=0.3, use_profile=False):
        super().__init__(env)
        self.validator = validator
        self.validation_weight = validation_weight
        self.use_profile = use_profile   # ✅ 추가

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if done:
            base_env = self.env.unwrapped
            outfit_ids = base_env.selected_ids

            if len(outfit_ids) >= 2:
                metadata = {
                    'gender': getattr(base_env, 'gender', 'All'),
                    'season': None
                }

                # ✅ train에서는 user_id=None으로 강제 → 개인화 X
                if self.use_profile:
                    user_ids = list(self.validator.user_logs.keys())
                    user_id = random.choice(user_ids) if user_ids else None
                else:
                    user_id = None

                val_score, accepted, details = self.validator.evaluate(
                    outfit_ids,
                    metadata,
                    user_id=user_id
                )

                original_reward = reward
                reward = (1 - self.validation_weight) * reward + self.validation_weight * val_score

                info.update({
                    "validation_score": val_score,
                    "validation_accepted": accepted,
                    "original_reward": original_reward,
                    "final_reward": reward,
                    "user_id": user_id
                })

                wandb.log({
                    "validation/score": val_score,
                    "validation/accepted": 1.0 if accepted else 0.0,
                })

        return obs, reward, done, truncated, info

# ===================================
# 환경 생성 함수
# ===================================
device = "cuda" if torch.cuda.is_available() else "cpu"
enc = EncoderWrapper(device=device)

def make_env():
    env = OutfitCompositionEnv(
        "./data_sources/fashion_products.pkl", 
        encoder=enc, 
        top_k=40
    )
    env = FashionEnvWrapper(env, prompts, gender="All")
    env = ExplorationWrapper(env, initial_epsilon=0.5)

    # ✅ 학습용 → 무조건 use_profile=False (개인화 X)
    env = ValidationRewardWrapper(env, validator,
                                  validation_weight=0.3,
                                  use_profile=False)

    env = FixedActionSpaceWrapper(env, max_actions=50)
    return env


print("🔄 Creating vectorized environment...")
vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# ===================================
# 진행률 콜백
# ===================================
class TqdmJsonCallback(BaseCallback):
    def __init__(self, total_timesteps: int, json_path: str, log_interval: int = 1000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.json_path = json_path
        self.log_interval = log_interval
        self.pbar = None
        self.logs = []

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training Progress",
            leave=True,
        )

    def _on_step(self) -> bool:
        self.pbar.update(1)

        if self.num_timesteps % self.log_interval == 0:
            log = {
                "total_timesteps": int(self.num_timesteps),
                "n_updates": int(getattr(self.model, "_n_updates", 0)),
            }
            if hasattr(self.model.logger, "name_to_value"):
                for key, value in self.model.logger.name_to_value.items():
                    log[key] = self._convert_to_json_serializable(value)
            self.logs.append(log)
        
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
        with open(self.json_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"\n✅ Logs saved to {self.json_path}")
    
    def _convert_to_json_serializable(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        else:
            return obj

# ===================================
# PPO 학습
# ===================================
total_timesteps = 200000

print("🔄 Initializing PPO model...")
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    batch_size=64,
    n_steps=512,
    gamma=0.99,
    learning_rate=1e-4,
    ent_coef=0.02,
    clip_range=0.1,
    gae_lambda=0.9,
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_epochs=10,
    device=device
)

# 로거 설정
new_logger = configure("./logs", ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# 초기 모델 저장
model.save("ppo_initial_personalized")
print("💾 Initial model saved")

# 콜백 생성
tqdm_callback = TqdmJsonCallback(
    total_timesteps=total_timesteps, 
    json_path="ppo_training_log_personalized.json"
)
wandb_callback = WandbCallback()

# 학습 시작
print(f"\n🚀 Starting training for {total_timesteps} timesteps...")
print(f"   Device: {device}")
print(f"   Validation enabled: True")
print(f"   Validation weight: 0.3")
print(f"   User profiles: {len(log_manager.logs)}")

model.learn(
    total_timesteps=total_timesteps,
    callback=[tqdm_callback, wandb_callback],
    progress_bar=True
)

# 학습 종료
print("\n✅ Training complete!")

# 모델 저장
model.save("ppo_fashion_model_personalized")
vec_env.save("vec_normalize_personalized.pkl")
print("💾 Final model saved:")
print("   - ppo_fashion_model_personalized.zip")
print("   - vec_normalize_personalized.pkl")

# WandB 종료
wandb.finish()
print("✅ All done!")