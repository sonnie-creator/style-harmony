# user_logs.py

import json
import pickle
from datetime import datetime

class UserLogManager:
    """사용자 로그 관리"""
    def __init__(self, log_file="user_logs.json"):
        self.log_file = log_file
        self.logs = self._load_logs()
    
    def _load_logs(self):
        """로그 파일 로드"""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_logs(self):
        """로그 파일 저장"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def add_like(self, user_id, item_id):
        """좋아요 추가"""
        if user_id not in self.logs:
            self.logs[user_id] = {
                'liked_items': [],
                'disliked_items': [],
                'preferred_styles': [],
                'disliked_colors': [],
                'purchase_history': []
            }
        
        if item_id not in self.logs[user_id]['liked_items']:
            self.logs[user_id]['liked_items'].append(item_id)
        
        # 싫어요에서 제거 (마음이 바뀐 경우)
        if item_id in self.logs[user_id]['disliked_items']:
            self.logs[user_id]['disliked_items'].remove(item_id)
        
        self.save_logs()
    
    def add_dislike(self, user_id, item_id):
        """싫어요 추가"""
        if user_id not in self.logs:
            self.logs[user_id] = {
                'liked_items': [],
                'disliked_items': [],
                'preferred_styles': [],
                'disliked_colors': [],
                'purchase_history': []
            }
        
        if item_id not in self.logs[user_id]['disliked_items']:
            self.logs[user_id]['disliked_items'].append(item_id)
        
        # 좋아요에서 제거
        if item_id in self.logs[user_id]['liked_items']:
            self.logs[user_id]['liked_items'].remove(item_id)
        
        self.save_logs()
    
    def add_purchase(self, user_id, item_id):
        """구매 이력 추가"""
        if user_id not in self.logs:
            self.logs[user_id] = {
                'liked_items': [],
                'disliked_items': [],
                'preferred_styles': [],
                'disliked_colors': [],
                'purchase_history': []
            }
        
        self.logs[user_id]['purchase_history'].append({
            'item_id': item_id,
            'timestamp': datetime.now().isoformat()
        })
        
        self.save_logs()
    
    def get_user_log(self, user_id):
        """사용자 로그 조회"""
        return self.logs.get(user_id, {})