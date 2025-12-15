from stable_baselines3 import PPO
from env_outfit_infer import OutfitInferenceEnv
import pickle
from huggingface_hub import hf_hub_download
model = PPO.load("model/ppo_fashion_model.zip")

with open("model/vec_normalize.pkl", "rb") as f:
    vec_normalize = pickle.load(f)
dataset_file = hf_hub_download(
    repo_id="Sonnie108/style-harmony",
    filename="dataset.pkl"
)

env = OutfitInferenceEnv(dataset_path=dataset_file)

def generate_recommendation(prompt: str, gender: str):
    obs, info = env.reset_env(prompt_text=prompt, gender=gender)

    done = False
    while not done:
        action, _ = model.predict(vec_normalize.normalize_obs(obs), deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    return env.selected_ids
