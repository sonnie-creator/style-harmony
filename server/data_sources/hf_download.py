from huggingface_hub import snapshot_download
import os

repo_id = "Sonnie108/style-harmony"
local_root = "./server/data_sources/images"

# 전체 repo snapshot을 로컬로 다운로드
local_repo_path = snapshot_download(repo_id=repo_id, cache_dir=local_root, repo_type="dataset")

print(f"Repo downloaded to: {local_repo_path}")
