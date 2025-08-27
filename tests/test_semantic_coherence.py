"""
Semantic coherence tests for the scoring model.
"""

import pandas as pd
import torch
from text_image_similarity.config import Config
from text_image_similarity.record_class import Record
from text_image_similarity.model_loader import get_model_and_processor
from text_image_similarity.scoring import score_pairs

# Preload model once for all tests in this module
model, processor = get_model_and_processor()


def resolve_device_config():
    """Resolves the device to use for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Config(
        batch_size=8, device=device, precision="fp32", timeout_s=10, retries=1
    )


cfg = resolve_device_config()


def test_paired_vs_shuffled_scores():
    """
    Tests that correctly paired images and captions score higher than shuffled pairs.
    """
    df = pd.read_csv("challenge_set.csv").sample(n=16, random_state=42)
    records = [Record(url=u, caption=c) for u, c in zip(df["url"], df["caption"])]

    true_scores = [r.score for r in score_pairs(records, model, processor, cfg)]

    shuffled_captions = df["caption"].sample(frac=1.0, random_state=7).tolist()
    mismatch_records = [
        Record(url=u, caption=c) for u, c in zip(df["url"], shuffled_captions)
    ]
    mismatch_scores = [
        r.score for r in score_pairs(mismatch_records, model, processor, cfg)
    ]

    mean_true = sum(true_scores) / len(true_scores)
    mean_mismatch = sum(mismatch_scores) / len(mismatch_scores)

    print(f"mean(true)={mean_true:.4f}  mean(shuffled)={mean_mismatch:.4f}")
    assert mean_true > mean_mismatch + 0.01, "Paired scores not > shuffled!"


def test_top_1_retrieval_on_mini_set():
    """
    Tests that for a distinctive set of images, each caption retrieves its correct image.
    """
    mini = pd.DataFrame(
        [
            {
                "url": "https://cdn.leonardo.ai/users/70eb2776-7757-4992-9980-55e780931fd8/generations/46cefe40-b107-4340-a73c-88a9af2455cb/Absolute_Reality_v16_A_photorealistic_of_Santa_Claus_walking_a_0.jpg",
                "caption": "A photorealistic of Santa Claus walking alone on the surface of the moon at night.",
            },
            {
                "url": "https://cdn.leonardo.ai/users/441e30e9-fe13-4f22-acaf-838135876ab7/generations/db00712b-deaf-463b-ba3a-759339dcf612/3D_Animation_Style_Cute_koala_bear_eating_bamboo_in_a_jungle_0.jpg",
                "caption": "Cute koala bear eating bamboo in a jungle",
            },
            {
                "url": "https://cdn.leonardo.ai/users/b81d5101-fb09-4410-9f47-b8b9ed1335df/generations/176b3af7-3cb7-497d-8d8f-3bd14a7e4c01/Default_A_pile_of_bricks_on_the_floor_2.jpg",
                "caption": "A pile of bricks on the floor",
            },
        ]
    )

    urls = mini["url"].tolist()
    caps = mini["caption"].tolist()

    from itertools import product
    import numpy as np

    pairs = [(i, j) for i, j in product(range(len(urls)), range(len(caps)))]
    recs = [Record(url=urls[i], caption=caps[j]) for i, j in pairs]
    scores = [r.score for r in score_pairs(recs, model, processor, cfg)]

    M = np.zeros((len(urls), len(caps)), dtype=float)
    for (i, j), s in zip(pairs, scores):
        M[i, j] = s

    top1 = M.argmax(axis=0)
    acc = (top1 == range(len(caps))).mean()
    print("Top-1 accuracy on mini-set:", acc)
    assert (
        acc == 1.0
    ), "Each caption should retrieve its matching image in this mini-set."


def test_caption_quality_ablation():
    """
    Tests that a more relevant caption scores higher than a less relevant one.
    """
    row = {
        "url": "https://cdn.leonardo.ai/users/70eb2776-7757-4992-9980-55e780931fd8/generations/46cefe40-b107-4340-a73c-88a9af2455cb/Absolute_Reality_v16_A_photorealistic_of_Santa_Claus_walking_a_0.jpg",
        "cap_full": "A photorealistic of Santa Claus walking alone on the surface of the moon at night...",
        "cap_short": "Santa Claus walking on the moon at night",
        "cap_wrong": "vector pattern pastel kawai cartoon repeating pattern",
    }

    recs = [
        Record(url=row["url"], caption=row["cap_full"]),
        Record(url=row["url"], caption=row["cap_short"]),
        Record(url=row["url"], caption=row["cap_wrong"]),
    ]

    scores = [r.score for r in score_pairs(recs, model, processor, cfg)]
    print("full, short, wrong =", scores)
    assert (
        scores[0] >= scores[1] > scores[2] - 1e-6
    ), "On-topic captions should score higher than wrong captions."


def test_stability_across_runs():
    """
    Tests that scores are deterministic across multiple runs.
    """
    df = pd.read_csv("challenge_set.csv").sample(n=8, random_state=42)
    recs = [Record(url=u, caption=c) for u, c in zip(df["url"], df["caption"])]

    s1 = [r.score for r in score_pairs(recs, model, processor, cfg)]
    s2 = [r.score for r in score_pairs(recs, model, processor, cfg)]

    delta = max(abs(a - b) for a, b in zip(s1, s2))
    print("max delta across runs:", delta)
    assert delta < 1e-6, "Scores should be stable across runs in eval/inference mode."
