import pickle
import pandas as pd

# ---------------------------------------------------------
# 1) Load raw pickle
# ---------------------------------------------------------
def load_raw_pkl(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return pd.DataFrame(data)


# ---------------------------------------------------------
# 2) Basic cleanup: gender / color / season / age_group
# ---------------------------------------------------------
def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gender"] = df["gender"].fillna("unknown")
    df["color"] = df["color"].fillna("unknown")
    df["season"] = df["season"].fillna("all")
    df["age_group"] = df["age_group"].fillna("all")

    return df


# ---------------------------------------------------------
# 3) Style group 및 Outer 재분류
# ---------------------------------------------------------
def refine_style(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # style_group을 기반으로 outer 재분류
    outer_map = {
        "jacket": "jacket",
        "coat": "coat",
        "cardigan": "cardigan",
        "windbreaker": "windbreaker",
        "padding": "padding",
    }

    df["outer_subtype"] = df["style_group"].map(outer_map).fillna("none")

    return df


# ---------------------------------------------------------
# 4) 불필요한 그룹 제거 (Unknown / Accessory / Set / Pack)
# ---------------------------------------------------------
def filter_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    remove_types = [ "acc", "set", "pack"]
    df = df[~df["product_type"].isin(remove_types)]
    df = df[df["product_type"] != "unknown"]

    return df


# ---------------------------------------------------------
# 5) Final dict 생성
# ---------------------------------------------------------
def to_dict(df: pd.DataFrame) -> dict:
    final_dict = {}
    for _, row in df.iterrows():
        final_dict[row["article_id"]] = {
            "product_name": row.get("product_name"),
            "product_type": row.get("product_type"),
            "gender": row.get("gender"),
            "color": row.get("color"),
            "season": row.get("season"),
            "age_group": row.get("age_group"),
            "style_group": row.get("style_group"),
            "outer_subtype": row.get("outer_subtype"),
            "clip_emb": row.get("clip_emb", None),
        }
    return final_dict


# ---------------------------------------------------------
# 6) Save final pickle
# ---------------------------------------------------------
def save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def run_pipeline(
    raw_pkl="raw_data.pkl",
    output_pkl="cleaned_final.pkl",
):
    print("Loading raw data...")
    df = load_raw_pkl(raw_pkl)

    print("Basic cleanup...")
    df = clean_basic(df)

    print("Refining style groups...")
    df = refine_style(df)

    print("Filtering items...")
    df = filter_items(df)

    print("Building final dictionary...")
    final_dict = to_dict(df)

    print(f"Saving to {output_pkl}...")
    save_pkl(final_dict, output_pkl)

    print("Done!")


if __name__ == "__main__":
    run_pipeline()

