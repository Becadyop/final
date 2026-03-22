"""
Echo Feeling - Sample Dataset Generator
Creates a synthetic reviews.csv for training and testing when no real dataset is available.
Also generates placeholder sticker folders with README files.

Usage:
    python generate_sample_data.py --out_dir data --n 3000
"""

import argparse
import random
import os
import csv
from pathlib import Path

random.seed(42)

# ── Template sentences per sentiment ─────────────────────────────────────────

POSITIVE_TEMPLATES = [
    "This product is absolutely amazing, I love it!",
    "Best purchase I have ever made. Highly recommend!",
    "Excellent quality and fast delivery. Very satisfied.",
    "Works perfectly. Exceeded my expectations completely.",
    "Outstanding product. Will definitely buy again.",
    "Super happy with this. Great value for money.",
    "Perfect condition, arrived on time. Love it!",
    "Fantastic product! My whole family loves it.",
    "Really good quality. Worth every penny.",
    "Incredible product. Five stars all the way!",
    "Loved the packaging and the product itself.",
    "Great item, exactly as described. 10/10.",
    "Quick shipping and the product is top notch.",
    "Amazing build quality. Very impressed.",
    "So glad I bought this. Life-changing product!",
]

NEGATIVE_TEMPLATES = [
    "Terrible product. Broke after one day of use.",
    "Complete waste of money. Do not buy this.",
    "Very disappointed. Nothing like the description.",
    "Poor quality. Fell apart immediately.",
    "Worst purchase ever. Absolutely horrible.",
    "Product stopped working within a week.",
    "Cheap material, terrible finish. Very unhappy.",
    "Would give zero stars if I could.",
    "Not as advertised at all. Total scam.",
    "Arrived damaged and customer service was useless.",
    "Feels flimsy and cheap. Very disappointed.",
    "Horrible experience from start to finish.",
    "Does not work at all. Returning immediately.",
    "Extremely poor quality for the price.",
    "Misleading product images. Very unsatisfied.",
]

NEUTRAL_TEMPLATES = [
    "Product is okay. Nothing special about it.",
    "Decent quality but could be better.",
    "Average product. Does what it says.",
    "It is fine, but I expected more.",
    "Not bad, not great. Just average.",
    "Works as described. Nothing to complain about.",
    "Reasonable quality for the price.",
    "Mediocre at best. Would not strongly recommend.",
    "It serves its purpose.",
    "Ordinary product. Average experience overall.",
    "Some good points, some not so good.",
    "Does the job but nothing impressive.",
    "Acceptable quality. Shipping was okay.",
    "It is what it is. Nothing special.",
    "Functional but uninspiring.",
]

SUSPICIOUS_TEMPLATES = [
    "This is the best product in the universe buy now!!!",
    "I was paid to review this. Amazing amazing amazing!",
    "DO NOT BUY THIS PRODUCT!!! SCAM SCAM SCAM!!!",
    "Fake fake fake product. Report this seller!!!",
    "Five stars because they gave me a discount code!!!",
    "WORST WORST WORST never buy from this store!!!",
    "Buy this now click here for discount!!! Best ever!!!",
    "One star because competitor product is way better!!!",
    "I got this for free in exchange for a 5 star review.",
    "Amazing product!!! Visit my website for more deals!!!",
]

# ── Emoji sets per sentiment ──────────────────────────────────────────────────

POSITIVE_EMOJIS = ["😊", "😍", "❤️", "👍", "🎉", "🤩", "😄", "💯", "✅", "🥰"]
NEGATIVE_EMOJIS = ["😡", "😢", "👎", "💔", "😤", "🤮", "😞", "😠", "🤬", "😭"]
NEUTRAL_EMOJIS  = ["😐", "🤔", "😶", "😑", ""]
SUSPICIOUS_EMOJIS = ["‼️", "⚠️", "🚨", "💰", "🤑"]

# ── Sticker label probabilities per sentiment ─────────────────────────────────

STICKER_DIST = {
    "positive"  : {"positive": 0.7, "neutral": 0.2, "negative": 0.1},
    "negative"  : {"negative": 0.7, "neutral": 0.2, "positive": 0.1},
    "neutral"   : {"neutral": 0.6,  "positive": 0.2, "negative": 0.2},
    "suspicious": {"negative": 0.5, "positive": 0.4, "neutral": 0.1},
}


def pick_sticker(label: str) -> str:
    dist = STICKER_DIST[label]
    keys, weights = zip(*dist.items())
    return random.choices(keys, weights=weights, k=1)[0]


def add_emojis(text: str, label: str, emoji_prob: float = 0.4) -> str:
    if random.random() > emoji_prob:
        return text
    pool = {
        "positive"  : POSITIVE_EMOJIS,
        "negative"  : NEGATIVE_EMOJIS,
        "neutral"   : NEUTRAL_EMOJIS,
        "suspicious": SUSPICIOUS_EMOJIS,
    }[label]
    n_emojis = random.randint(1, 3)
    emojis = "".join(random.choices(pool, k=n_emojis))
    return text + " " + emojis


def generate_dataset(n: int = 3000) -> list[dict]:
    """
    Generate n synthetic reviews.
    Distribution roughly matches the paper: ~88% positive, ~8% negative, ~4% neutral.
    """
    target_dist = {
        "positive"  : int(n * 0.55),
        "negative"  : int(n * 0.20),
        "neutral"   : int(n * 0.15),
        "suspicious": int(n * 0.10),
    }
    # Adjust for rounding
    total = sum(target_dist.values())
    target_dist["positive"] += n - total

    templates = {
        "positive"  : POSITIVE_TEMPLATES,
        "negative"  : NEGATIVE_TEMPLATES,
        "neutral"   : NEUTRAL_TEMPLATES,
        "suspicious": SUSPICIOUS_TEMPLATES,
    }

    rows = []
    for label, count in target_dist.items():
        for _ in range(count):
            base  = random.choice(templates[label])
            text  = add_emojis(base, label)
            sticker = pick_sticker(label)
            rows.append({"review": text, "label": label, "sticker": sticker})

    random.shuffle(rows)
    return rows


def create_sticker_folders(root: Path) -> None:
    """Create positive/negative/neutral sticker directories with README placeholders."""
    for folder in ["positive", "negative", "neutral"]:
        fpath = root / folder
        fpath.mkdir(parents=True, exist_ok=True)
        readme = fpath / "README.txt"
        if not readme.exists():
            readme.write_text(
                f"Place {folder}-sentiment sticker PNG images in this folder.\n"
                "Each PNG will be labelled as '{folder}' sentiment.\n"
                "Example sticker counts used in the paper:\n"
                "  positive: 25 stickers\n"
                "  negative: 22 stickers\n"
                "  neutral : 11 stickers\n"
            )
    print(f"[INFO] Sticker folders created at: {root}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample Echo Feeling dataset")
    parser.add_argument("--out_dir", default="data",  help="Output directory")
    parser.add_argument("--n",       type=int, default=3000, help="Number of reviews to generate")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate CSV
    rows = generate_dataset(args.n)
    csv_path = out / "reviews.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["review", "label", "sticker"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote {len(rows)} reviews to {csv_path}")

    # Distribution summary
    from collections import Counter
    dist = Counter(r["label"] for r in rows)
    for label, count in sorted(dist.items()):
        print(f"  {label:12s}: {count:4d} ({count/len(rows)*100:.1f}%)")

    # Create sticker folders
    create_sticker_folders(out / "stickers")

    print(f"\n[DONE] Dataset ready. Now run:")
    print(f"  python training_phase/train.py --data_path {csv_path} --sticker_root {out / 'stickers'}")


if __name__ == "__main__":
    main()
