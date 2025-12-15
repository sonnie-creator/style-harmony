# env_outfit.py 
import os
import pickle
import random
from typing import List, Dict, Optional, Tuple
from PIL import Image
import gymnasium as gym
import numpy as np

class EncoderWrapper:
    """Fashion-CLIP 기반 텍스트/이미지 임베더 래퍼"""
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._loaded = False

    def load_fashion_clip(self):
        try:
            import open_clip
        except Exception as e:
            raise RuntimeError(
                "open_clip library not available. "
                "Install with `pip install open-clip-torch`"
            ) from e

        model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:Marqo/marqo-fashionCLIP', 
            pretrained=True
        )
        tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')

        model.eval()
        model.to(self.device)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self._loaded = True

    def encode_text(self, texts: List[str]) -> np.ndarray:
        if not self._loaded:
            self.load_fashion_clip()

        import torch
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            arr = text_features.cpu().numpy()
        return arr

    def encode_image_paths(self, paths: List[str], batch_size: int = 16) -> np.ndarray:
        if not self._loaded:
            self.load_fashion_clip()

        import torch

        out_embs = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            imgs = [self.preprocess(Image.open(p).convert("RGB")) for p in batch]
            imgs = torch.stack(imgs).to(self.device)
            with torch.no_grad():
                img_features = self.model.encode_image(imgs)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                out_embs.append(img_features.cpu().numpy())
        return np.vstack(out_embs)


CATEGORY_ORDER = ["TopOrDress", "Bottom", "Outer", "Shoes", "Accessories"]

class OutfitCompositionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 dataset_pkl: str,
                 encoder: Optional[EncoderWrapper] = None,
                 top_k: int = 50,
                 embedding_key: str = "clip_emb",
                 device: str = "cpu",
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 max_actions: int = 100,
                 use_user_log_reward: bool = True):
        super().__init__()
        self.device = device
        self.top_k = top_k
        self.embedding_key = embedding_key
        self.alpha = alpha
        self.beta = beta
        self.max_actions = max_actions
        self.use_user_log_reward = use_user_log_reward

        with open(dataset_pkl, "rb") as f:
            self.dataset: Dict = pickle.load(f)
        
        if isinstance(self.dataset, list):
            self.dataset = {item["article_id"]: item for item in self.dataset}

# 🔥 모든 key를 문자열로 강제 변환 (KeyError 해결 핵심)
        self.dataset = {str(k): v for k, v in self.dataset.items()}
        self.encoder = encoder or EncoderWrapper(device=self.device)

        self.items_by_cat = {
            "Top": [], "Dress": [], "Bottom": [], "Outer": [], "Shoes": [], "Accessories": []
        }
        
        for item_id, info in self.dataset.items():
            cat = info.get("style_type")
            article_type = info.get("prod_name", "").lower()
            if cat == "Bottom" and any(x in article_type for x in ["tights", "leggings", "stockings", "socks"]):
                continue
            if cat in self.items_by_cat:
                self.items_by_cat[cat].append(item_id)

        self._ensure_item_embeddings()

        self.prompt_emb: Optional[np.ndarray] = None
        self.selected_ids: List = []
        self.selected_embs: List[np.ndarray] = []
        self.selected_categories: List[str] = []
        self.current_step = 0
        self.candidate_pool_for_step: List = []
        self.valid_action_mask: Optional[np.ndarray] = None
        self.chose_dress = False
        self.season: Optional[str] = None

        emb_dim = next(iter(self.dataset.values()))[self.embedding_key].shape[0]
        self.emb_dim = emb_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(emb_dim*2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.max_actions)

    def _ensure_item_embeddings(self):
        import torch

        missing = [
            iid for iid, info in self.dataset.items()
            if self.embedding_key not in info or info[self.embedding_key] is None
        ]
        if not missing:
            return

        if not self.encoder._loaded:
            self.encoder.load_fashion_clip()

        model_device = next(self.encoder.model.parameters()).device

        for i in range(0, len(missing), 32):
            batch_ids = missing[i:i+32]
            batch_imgs = []
            for iid in batch_ids:
                path = self.dataset[iid].get("image_path")
                if path is None or not isinstance(path, str):
                    continue
                batch_imgs.append(self.encoder.preprocess(Image.open(path).convert("RGB")))

            if not batch_imgs:
                continue

            batch_tensor = torch.stack(batch_imgs).to(model_device)

            with torch.no_grad():
                batch_emb = self.encoder.model.encode_image(batch_tensor)
                batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
                batch_emb = batch_emb.cpu().numpy().astype("float32")

            for j, iid in enumerate(batch_ids):
                self.dataset[iid]["clip_emb"] = batch_emb[j]

    def _infer_style_from_prompt(self, prompt_text: str) -> List[str]:
        style_descriptions = {
            "casual": "casual everyday clothing, comfortable and relaxed style",
            "formal": "formal business attire, elegant and professional clothing",
            "sporty": "sporty athletic wear, active sportswear",
            "ethnic": "ethnic traditional clothing, cultural style",
            "party": "party festive clothing, glamorous outfit",
            "travel": "travel comfortable clothing, practical outfit",
            "home": "home casual wear, comfortable loungewear"
        }
        prompt_emb = self.encoder.encode_text([prompt_text])[0]
        style_embs = self.encoder.encode_text(list(style_descriptions.values()))
        similarities = (style_embs / np.linalg.norm(style_embs, axis=1, keepdims=True)) @ (prompt_emb / (np.linalg.norm(prompt_emb)+1e-8))
        top_idx = np.argsort(-similarities)[:2]
        return [list(style_descriptions.keys())[i] for i in top_idx]

    def _infer_color_from_prompt(self, prompt_text: str) -> Optional[str]:
        color_descriptions = {
            "black_white_grey": "black white grey neutral monochrome",
            "warm": "warm red orange yellow brown vibrant",
            "cool": "cool blue green navy calming",
            "pastel": "pastel soft pink cream lavender",
            "metallic": "metallic silver gold bronze shiny",
            "multi": "multicolored pattern mixed colors"
        }
        prompt_emb = self.encoder.encode_text([prompt_text])[0]
        color_embs = self.encoder.encode_text(list(color_descriptions.values()))
        similarities = (color_embs / np.linalg.norm(color_embs, axis=1, keepdims=True)) @ (prompt_emb / (np.linalg.norm(prompt_emb)+1e-8))
        best_idx = np.argmax(similarities)
        return list(color_descriptions.keys())[best_idx] if similarities[best_idx] > 0.35 else None

    def reset(self, prompt_text: str, gender: str, age: Optional[str] = None,
              personal_color: Optional[str] = None, season: Optional[str] = None,
              seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_prompt = prompt_text
        self.user_gender = gender
        self.user_age = age
        self.personal_color_group = personal_color
        self.season = season
        self.target_gender = gender

        inferred_style = self._infer_style_from_prompt(prompt_text)
        inferred_color = self._infer_color_from_prompt(prompt_text)

        self.priority_styles = inferred_style if inferred_style else []
        self.target_color_group = inferred_color if inferred_color else personal_color

        self.prompt_emb = self.encoder.encode_text([prompt_text])[0].astype(np.float32)

        self.selected_ids = []
        self.selected_embs = []
        self.selected_categories = []
        self.current_step = 0
        self.chose_dress = False
        
        self._update_candidate_pool()

        obs = self._get_observation()
        info = {
            "priority_styles": self.priority_styles,
            "target_color_group": self.target_color_group,
            "user_age": age,
            "personal_color": personal_color,
            "season": season,
            "gender": gender,
        }
        return obs, info
    def _get_observation(self) -> np.ndarray:
        if len(self.selected_embs) == 0:
            agg = np.zeros((self.emb_dim,), dtype=np.float32)
        else:
            agg = np.mean(np.stack(self.selected_embs, axis=0), axis=0).astype(np.float32)
        return np.concatenate([self.prompt_emb, agg], axis=0)

    def _update_candidate_pool(self):
        """
        ✅ 개선된 후보 업데이트:
        1. TopOrDress → Top/Dress 중 선택
        2. Dress 선택 시 Bottom 스킵
        """
        if self.current_step >= len(CATEGORY_ORDER):
            self.candidate_pool_for_step = []
            self.valid_action_mask = np.zeros(self.max_actions, dtype=bool)
            return

        cat = CATEGORY_ORDER[self.current_step]
        
        # ✅ Top/Dress 선택 로직
        if cat == "TopOrDress":
            # 프롬프트에서 "dress" 언급 확인
            prompt_lower = self.current_prompt.lower()
            if "dress" in prompt_lower or "onepiece" in prompt_lower:
                all_ids = self.items_by_cat.get("Dress", [])
                actual_cat = "Dress"
            else:
                all_ids = self.items_by_cat.get("Top", [])
                actual_cat = "Top"
        
        # ✅ Dress 선택 시 Bottom 스킵
        elif cat == "Bottom" and self.chose_dress:
            print("✅ Dress selected, skipping Bottom")
            self.current_step += 1
            self._update_candidate_pool()
            return
        
        else:
            all_ids = self.items_by_cat.get(cat, [])
            actual_cat = cat

        if len(all_ids) == 0:
            print(f"⚠️ No items in {cat}")
            self.candidate_pool_for_step = []
            self.valid_action_mask = np.zeros(self.max_actions, dtype=bool)
            return

        # 프롬프트 유사도 기반 Top-K
        item_embs = np.stack([self.dataset[iid][self.embedding_key] for iid in all_ids], axis=0)
        pe = self.prompt_emb / (np.linalg.norm(self.prompt_emb) + 1e-8)
        item_norms = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-8)
        sims = (item_norms @ pe).astype(np.float32)

        k = min(self.top_k, len(all_ids))
        top_idx = np.argsort(-sims)[:k]
        
        # Diversity sampling
        split = int(k * 0.7)
        deterministic = [all_ids[int(i)] for i in top_idx[:split]]
        remaining = [all_ids[int(i)] for i in top_idx[split:]]
        random_count = k - split
        
        if len(remaining) > 0:
            random_part = random.sample(remaining, min(random_count, len(remaining)))
        else:
            random_part = []
        
        self.candidate_pool_for_step = deterministic + random_part
        
        num_candidates = len(self.candidate_pool_for_step)
        self.valid_action_mask = np.zeros(self.max_actions, dtype=bool)
        self.valid_action_mask[:num_candidates] = True
        
        print(f"✅ {actual_cat}: {num_candidates} candidates (top sim: {sims[top_idx[0]]:.3f})")
    
    def step(self, action: int):
        num_candidates = len(self.candidate_pool_for_step)
        
        # Invalid action 처리
        if action >= num_candidates or action < 0:
            print(f"⚠️ Invalid action {action}")
            
            if num_candidates > 0:
                action = random.randint(0, num_candidates - 1)
            else:
                self.current_step += 1
                done = self.current_step >= len(CATEGORY_ORDER)
                
                if not done:
                    self._update_candidate_pool()
                
                obs = self._get_observation()
                return obs, -0.5, done, False, {"skipped": True}
        
        chosen_id = self.candidate_pool_for_step[int(action)]
        chosen_emb = self.dataset[chosen_id][self.embedding_key].astype(np.float32)

        self.selected_ids.append(chosen_id)
        self.selected_embs.append(chosen_emb)
        
        # ✅ 카테고리 추적
        cat = CATEGORY_ORDER[self.current_step]
        if cat == "TopOrDress":
            actual_cat = self.dataset[chosen_id].get("style_type")
            if actual_cat == "Dress":
                self.chose_dress = True
            self.selected_categories.append(actual_cat)
        else:
            self.selected_categories.append(cat)

        self.current_step += 1
        done = self.current_step >= len(CATEGORY_ORDER)

        if not done:
            self._update_candidate_pool()

        reward = self._calc_reward(done)
        obs = self._get_observation()
        
        return obs, float(reward), bool(done), False, {"selected_ids": list(self.selected_ids)}
    def _calc_reward(self, done: bool) -> float:
        """보상 계산: prompt, embedding, 색상 조화/단색/클래식/원색/Lightness 반영"""
        pn = self.prompt_emb / (np.linalg.norm(self.prompt_emb) + 1e-8)

        # Step reward
        if not done:
            last_emb = self.selected_embs[-1]
            ln = last_emb / (np.linalg.norm(last_emb) + 1e-8)

            # Prompt 유사도
            prompt_sim = float(np.dot(pn, ln))
            step_reward = 0.7 * prompt_sim

            # Compatibility
            if len(self.selected_embs) > 1:
                prev = np.stack(self.selected_embs[:-1], axis=0)
                prev_norm = prev / (np.linalg.norm(prev, axis=1, keepdims=True) + 1e-8)
                comp_sims = np.dot(prev_norm, ln)
                comp_reward = float(np.mean(comp_sims))
                step_reward += 0.3 * comp_reward

            # 색상 관련 보상 계산
            color_groups = []
            lightness = []
            for iid in self.selected_ids:
                item = self.dataset[iid]
                color_groups.append(item.get("colour_group_name", "unknown"))
                lightness.append(item.get("lightness", 0.5))  # 기본값 0.5

            unique_colors = set(color_groups)
            primary_colors = {"red", "yellow", "blue"}
            classic_colors = {"black", "white", "denim"}

            # 단색/조화
            if len(unique_colors) <= 2:
                step_reward += 0.20
            if any(c in primary_colors for c in unique_colors):
                step_reward -= 0.15
            if all(c in classic_colors for c in unique_colors):
                step_reward += 0.15
            if len(unique_colors) > 3:
                step_reward -= 0.15

            # Lightness 대비
            for i in range(len(lightness)):
                for j in range(i + 1, len(lightness)):
                    li, lj = lightness[i], lightness[j]
                    ci, cj = color_groups[i], color_groups[j]
                    if {ci, cj}.issubset(classic_colors):
                        continue
                    if abs(li - lj) > 0.5:
                        step_reward *= 0.9

            return float(step_reward)

        # Final reward
        embs = np.stack(self.selected_embs, axis=0)
        norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        sim_matrix = norms @ norms.T
        m = embs.shape[0]

        if m <= 1:
            compatibility = 0.0
        else:
            tri_idx = np.triu_indices(m, k=1)
            compatibility = float(np.mean(sim_matrix[tri_idx]))

        full_prompt = float(np.mean([np.dot(pn, e) for e in norms]))
        final_reward = 1.2 * compatibility + 0.8 * full_prompt

        # Final reward 색상 보정
        primary_colors = {"red", "yellow", "blue"}
        classic_colors = {"black", "white", "denim"}

        color_groups = []
        lightness = []
        for iid in self.selected_ids:
            item_id_str = str(iid)
            item = self.dataset[item_id_str]
            color_groups.append(item.get("colour_group_name", "unknown"))
            lightness.append(item.get("lightness", 0.5))

        unique_colors = set(color_groups)

        if len(unique_colors) <= 2:
            final_reward *= 1.2
        if any(c in primary_colors for c in unique_colors):
            final_reward *= 0.7
        if all(c in classic_colors for c in unique_colors):
            final_reward *= 1.3
        if len(unique_colors) > 3:
            final_reward *= 0.8

        for i in range(len(lightness)):
            for j in range(i + 1, len(lightness)):
                li, lj = lightness[i], lightness[j]
                ci, cj = color_groups[i], color_groups[j]
                if {ci, cj}.issubset(classic_colors):
                    continue
                if abs(li - lj) > 0.5:
                    final_reward *= 0.9

        return float(final_reward)


    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Selected: {self.selected_categories}")

    def get_action_mask(self) -> np.ndarray:
        return self.valid_action_mask if self.valid_action_mask is not None else np.zeros(self.max_actions, dtype=bool)


class OutfitBatchRecommender:
    """
    ✨ PPO 모델을 사용해 완성된 코디 3벌을 생성하는 래퍼
    - 기존 OutfitCompositionEnv를 3번 실행
    - 다양성을 위해 Top-K 샘플링 적용
    """

    def __init__(self, env, model, vec_env, dataset):
        self.env = env
        self.model = model
        self.vec_env = vec_env
        self.dataset = dataset

    
    def recommend_outfits(self, prompt: str, gender: str = None, age: str = None, season: str = None,
                     personal_color: str = None, num_outfits: int = 3) -> List[Dict]:
        import sys
        """
        완성된 코디 num_outfits개 생성

        Returns:
            [
                {
                    'outfit_id': 1,
                    'items': {'top': {...}, 'bottom': {...}, ...},
                    'score': 0.85,
                    'categories': ['Top', 'Bottom', 'Shoes']
                },
                ...
            ]
        """
        print(f"\n{'='*60}")
        print(f"🎨 {num_outfits}개 코디 생성 시작")
        print(f"{'='*60}")

        outfits = []

        for outfit_idx in range(num_outfits):
            print(f"\n--- 코디 #{outfit_idx + 1} 생성 중 ---")

            # 환경 초기화
            base_env = self.vec_env.envs[0]
            obs_raw, info = base_env.reset(
                prompt_text=prompt,
                gender=gender,
                age=age,
                season=season,
                personal_color=personal_color
            )

            obs = self.vec_env.normalize_obs(obs_raw)
            obs = obs.reshape(1, -1)

            selected_items = {}
            step_rewards = []

            # 각 스텝별로 아이템 선택
            for step_idx in range(5):
                num_candidates = len(base_env.candidate_pool_for_step)
                if num_candidates == 0:
                    continue

                # 다양성을 위한 Top-K 샘플링
                action, _ = self.model.predict(obs, deterministic=True)
                if outfit_idx == 0:
                    action_int = min(int(action[0]), num_candidates - 1)
                elif outfit_idx == 1:
                    top_k = min(3, num_candidates)
                    action_int = random.randint(0, top_k - 1)
                else:
                    top_k = min(5, num_candidates)
                    action_int = random.randint(0, top_k - 1)

                # Step 실행
                obs_raw, reward, done, truncated, info_step = base_env.step(action_int)
                obs = self.vec_env.normalize_obs(obs_raw)
                obs = obs.reshape(1, -1)

                step_rewards.append(float(reward))

            # 각 아이템별로 dataset에서 조회
            for step_i, item_id in enumerate(base_env.selected_ids):
                article_id = str(item_id)  # ⭐ 문자열로 변환
                item = self.dataset.get(article_id)
                if item is None:
                    print(f"⚠️ Dataset에 {article_id} 없음", file=sys.stderr, flush=True)
                    continue
                cat = base_env.selected_categories[step_i].lower() if step_i < len(base_env.selected_categories) else f"item_{step_i}"
                selected_items[cat] = {
                    'id': article_id,
                    'article_id': article_id,
                    'name': item.get('prod_name', 'Unknown'),
                    'image_path': item.get('image_path', ''),
                    'style': item.get('product_type_name', 'N/A'),
                    'color': item.get('colour_group_name', 'N/A'),
                    'reward': step_rewards[step_i] if step_i < len(step_rewards) else 0.0
                }

            # 최종 보상 계산
            final_reward = base_env._calc_reward(done=True)

            # Compatibility 계산
            if len(base_env.selected_embs) > 1:
                embs = np.stack(base_env.selected_embs)
                norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
                sim_matrix = norms @ norms.T
                tri_idx = np.triu_indices(len(embs), k=1)
                compatibility = float(np.mean(sim_matrix[tri_idx]))
            else:
                compatibility = 0.0

            # Prompt match 계산
            prompt_emb = base_env.prompt_emb
            prompt_norm = prompt_emb / (np.linalg.norm(prompt_emb) + 1e-8)
            prompt_match = float(np.mean([
                np.dot(prompt_norm, emb / (np.linalg.norm(emb) + 1e-8))
                for emb in base_env.selected_embs
            ])) if base_env.selected_embs else 0.0

            # Reasoning 생성
            reasoning_parts = []
            if base_env.priority_styles:
                reasoning_parts.append(
                    f"스타일 '{', '.join(base_env.priority_styles)}' 유사도 우선 고려"
                )
            if base_env.target_color_group:
                reasoning_parts.append(
                    f"개인 색상 그룹 '{base_env.target_color_group}'과의 색채 조화"
                )

            for step_i, item_id in enumerate(base_env.selected_ids):
                article_id = str(item_id)
                item = self.dataset.get(article_id)
                cat = base_env.selected_categories[step_i] if step_i < len(base_env.selected_categories) else f"item_{step_i}"
                reward_i = step_rewards[step_i] if step_i < len(step_rewards) else 0.0
                reasoning_parts.append(
                    f"{cat}({item.get('prod_name', 'Unknown')}) - 유사도 및 조화 점수 {reward_i:.3f}"
                )

            reasoning_text = " → ".join(reasoning_parts) if reasoning_parts else "데이터 기반 선택"

            outfit = {
                'outfit_id': outfit_idx + 1,
                'items': selected_items,
                'categories': base_env.selected_categories,
                'reasoning': reasoning_text,
                'scores': {
                    'total': float(final_reward),
                    'compatibility': float(compatibility),
                    'prompt_match': float(prompt_match),
                    'step_rewards': step_rewards
                }
            }

            outfits.append(outfit)

            print(f"✅ 코디 #{outfit_idx + 1} 완료")
            print(f"   카테고리: {base_env.selected_categories}")
            print(f"   최종 점수: {final_reward:.3f}")

        print(f"\n{'='*60}")
        print(f"🎉 {len(outfits)}개 코디 생성 완료!")
        print(f"{'='*60}\n")

        return outfits


import json
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
class UserLogManager:
    def __init__(self, base_dir="./data_sources/user_json/"):
        self.base_dir = base_dir
    
    def _load_all(self):
        """🔄 매번 최신 파일 읽기"""
        logs = {}
        if not os.path.exists(self.base_dir):
            return logs
        
        for fname in os.listdir(self.base_dir):
            if fname.endswith(".json"):
                user_id = fname.replace(".json", "")
                try:
                    with open(os.path.join(self.base_dir, fname), "r", encoding="utf-8") as f:
                        logs[user_id] = json.load(f)
                except Exception as e:
                    print(f"⚠️ Failed to load {fname}: {e}")
                    continue
        return logs
    
    def get_user_data(self, user_id: str) -> Dict:
        """✅ 항상 최신 데이터 반환"""
        all_logs = self._load_all()
        return all_logs.get(user_id, {})


class ValidationAgent:
    """
    완성된 outfit 검증 + 사용자 개인화 (실시간 반영)
    """
    def __init__(self, dataset, user_log_manager: Optional[UserLogManager] = None, min_score: float = 0.4):
        self.dataset = dataset
        self.user_log_manager = user_log_manager  # ✅ Manager 저장
        self.min_score = min_score

    def _get_user_data(self, user_id: Optional[str]) -> Dict:
        """
        ✅ 매번 최신 사용자 로그 읽기
        """
        if not user_id or not self.user_log_manager:
            return {}
        
        return self.user_log_manager.get_user_data(user_id)
    
    def _extract_article_ids_from_feedback(self, feedback_list: List[Dict]) -> List[str]:
        """
        liked_items / disliked_items 안의 feedback_data에서 article_id만 추출
        """
        article_ids = []

        for fb in feedback_list:
            outfit_items = fb.get("outfit_items", {})
            if isinstance(outfit_items, dict):
                for item in outfit_items.values():
                    if isinstance(item, dict) and "article_id" in item:
                        article_ids.append(str(item["article_id"]))

        return article_ids
    
    def _extract_user_preferences(self, user_data: Dict) -> Dict:
        if not user_data:
            return {
                "liked_items": [],
                "disliked_items": [],
                "preferred_styles": [],
                "disliked_colors": []
            }

        preferences = user_data.get("preferences", {})

        liked_feedback = preferences.get("liked_items", [])
        disliked_feedback = preferences.get("disliked_items", [])

        return {
            "liked_items": self._extract_article_ids_from_feedback(liked_feedback),
            "disliked_items": self._extract_article_ids_from_feedback(disliked_feedback),
            "preferred_styles": preferences.get("preferred_styles", []),
            "disliked_colors": preferences.get("disliked_colors", [])
        }
        
    def evaluate(self, outfit_items: List[str], metadata: Dict, user_id: Optional[str] = None) -> Tuple[float, bool, Dict]:
        """
        Args:
            outfit_items: [article_id1, article_id2, ...]
            metadata: {'gender': 'Women', 'season': 'Spring', ...}
            user_id: 사용자 ID
        
        Returns:
            (score, accepted, details)
        """
        # ✅ 최신 사용자 데이터 로드
        if user_id:
            user_data = self._get_user_data(user_id)
            user_prefs = self._extract_user_preferences(user_data)
            
            print(f"[DEBUG] 로드된 사용자 데이터:")
            print(f"  - liked_items: {len(user_prefs['liked_items'])}개")
            print(f"  - disliked_items: {len(user_prefs['disliked_items'])}개")
        else:
            user_prefs = None
        
        # 개인화 검증
        if user_prefs and (user_prefs['liked_items'] or user_prefs['disliked_items']):
            scores = {
                'avoid_disliked': self._check_avoid_disliked(outfit_items, user_prefs),
                'style_match': self._check_style_match(outfit_items, user_prefs),
            }
            
            total_score = sum(scores.values()) / len(scores)
            accepted = total_score >= self.min_score
            
            print(f"[INFO] 개인화 검증: {user_id} - 점수: {total_score:.2f}")
            print(f"[INFO] 상세: {scores}")
            
            return total_score, accepted, scores
        else:
            print(f"[INFO] 사용자 로그 없음 → 기본 승인")
            return 1.0, True, {'default': 1.0}
    
    def _check_avoid_disliked(self, item_ids: List[str], user_prefs: Dict) -> float:
        """싫어요한 아이템이 포함되어 있는지 확인"""
        disliked_items = set(user_prefs['disliked_items'])
        
        if not disliked_items:
            return 1.0
        
        # 하나라도 포함되면 0점
        for item_id in item_ids:
            if item_id in disliked_items:
                print(f"[WARNING] 싫어요 아이템 포함: {item_id}")
                return 0.0
        
        return 1.0
    
    def _check_style_match(self, item_ids: List[str], user_prefs: Dict) -> float:
        """선호 스타일과 일치하는지 확인"""
        preferred_styles = user_prefs['preferred_styles']
        
        if not preferred_styles:
            return 1.0
        
        outfit_styles = []
        for item_id in item_ids:
            if item_id in self.dataset:
                style = self.dataset[item_id].get('season_style', '').lower()
                if style:
                    outfit_styles.append(style)
        
        if not outfit_styles:
            return 0.5
        
        match_count = sum(1 for s in outfit_styles if s in preferred_styles)
        score = match_count / len(outfit_styles)
        
        print(f"[INFO] 스타일 매칭: {match_count}/{len(outfit_styles)} = {score:.2f}")
        return score
    