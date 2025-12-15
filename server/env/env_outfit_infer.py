# outfit_infer.py
import numpy as np
from typing import List, Dict, Optional
import gymnasium as gym

from env.env_outfit_train import (
    EncoderWrapper,
    CATEGORY_ORDER,
)


class OutfitInferenceEnv(gym.Env):
    """
    추론(Recommendation) 전용 Environment.
    - 학습용 OutfitCompositionEnv보다 훨씬 단순화
    - reward 없음
    - step()은 선택만 수행하고 done 반환
    - observation 구조는 동일하게 유지 (prompt_emb + mean(selected_embs))
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dataset: Dict,
        encoder: EncoderWrapper,
        embedding_key: str = "clip_emb",
        top_k: int = 50,
        device: str = "cpu",
        max_actions: int = 100,
    ):
        super().__init__()
        self.dataset = {str(k): v for k, v in dataset.items()}
        self.encoder = encoder
        self.embedding_key = embedding_key
        self.top_k = top_k
        self.device = device
        self.max_actions = max_actions

        # Dataset에서 임베딩 차원 자동 감지
        first = next(iter(self.dataset.values()))
        self.emb_dim = first[self.embedding_key].shape[0]

        # PPO에서 요구하는 observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.emb_dim * 2,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.max_actions)

    """
    =============================
      RESET
    =============================
    """

    def reset(
        self,
        prompt_text: str,
        gender: Optional[str] = None,
        season: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        # Prompt
        self.prompt_text = prompt_text
        self.prompt_emb = self.encoder.encode_text([prompt_text])[0].astype(np.float32)

        # Attributes
        self.gender = gender
        self.season = season

        # State
        self.selected_ids: List[str] = []
        self.selected_embs: List[np.ndarray] = []
        self.current_step = 0

        # Items by category (학습 env의 규칙과 동일하게 구성)
        self._build_items_by_category()

        # 첫 Step 후보 생성
        self._update_candidate_pool()

        # Observation 반환
        return self._get_observation(), {}

    """
    =============================
      내부 구성 함수
    =============================
    """

    def _build_items_by_category(self):
        self.items_by_cat = {"Top": [], "Dress": [], "Bottom": [], "Outer": [], "Shoes": [], "Accessories": []}

        for iid, info in self.dataset.items():
            cat = info.get("style_type")
            if cat in self.items_by_cat:
                self.items_by_cat[cat].append(iid)

    def _update_candidate_pool(self):
        if self.current_step >= len(CATEGORY_ORDER):
            self.candidate_pool_for_step = []
            return

        cat = CATEGORY_ORDER[self.current_step]

        # TopOrDress 처리
        if cat == "TopOrDress":
            if "dress" in self.prompt_text.lower():
                all_ids = self.items_by_cat.get("Dress", [])
                self.actual_category = "Dress"
            else:
                all_ids = self.items_by_cat.get("Top", [])
                self.actual_category = "Top"

        # Dress → Bottom 스킵
        elif cat == "Bottom" and "Dress" in self.selected_categories:
            self.current_step += 1
            self._update_candidate_pool()
            return

        else:
            all_ids = self.items_by_cat.get(cat, [])
            self.actual_category = cat

        if len(all_ids) == 0:
            self.candidate_pool_for_step = []
            return

        # Prompt similarity 기반 Top-K
        item_embs = np.stack([self.dataset[i][self.embedding_key] for i in all_ids], axis=0)
        pe = self.prompt_emb / (np.linalg.norm(self.prompt_emb) + 1e-8)
        norms = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-8)
        sims = norms @ pe

        k = min(self.top_k, len(all_ids))
        top_idx = np.argsort(-sims)[:k]
        self.candidate_pool_for_step = [all_ids[i] for i in top_idx]

    def _get_observation(self):
        if len(self.selected_embs) == 0:
            mean_emb = np.zeros((self.emb_dim,), dtype=np.float32)
        else:
            mean_emb = np.mean(np.stack(self.selected_embs, axis=0), axis=0).astype(np.float32)

        return np.concatenate([self.prompt_emb, mean_emb], axis=0)

    """
    =============================
      STEP
    =============================
    """

    def step(self, action: int):
        if len(self.candidate_pool_for_step) == 0:
            done = True
            return self._get_observation(), 0.0, done, False, {"empty": True}

        # action clip
        if action >= len(self.candidate_pool_for_step):
            action = 0

        chosen_id = self.candidate_pool_for_step[action]
        chosen_emb = self.dataset[chosen_id][self.embedding_key].astype(np.float32)

        self.selected_ids.append(chosen_id)
        self.selected_embs.append(chosen_emb)
        self.selected_categories = [self.actual_category]

        # Next category
        self.current_step += 1
        done = self.current_step >= len(CATEGORY_ORDER)

        if not done:
            self._update_candidate_pool()

        return self._get_observation(), 0.0, done, False, {"chosen": chosen_id}

