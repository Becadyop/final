"""
Echo Feeling - Deployment Server (Flask)
Exposes REST endpoints consumed by the e-commerce admin panel and the Node.js wrapper.

Endpoints:
  POST /analyze           – analyse a single review
  POST /analyze/batch     – analyse a list of reviews
  GET  /product/<id>      – product-level sentiment summary
  POST /product/<id>/add  – add a review and re-compute summary
  DELETE /review/<id>     – flag/delete a suspicious review
  GET  /health            – health-check
"""

import json
import uuid
from pathlib import Path
from collections import defaultdict

from flask import Flask, request, jsonify, abort
from flask_cors import CORS

# ── Echo Feeling engine ───────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from echo_feeling.api import analyze, analyze_batch, product_dashboard

# ── In-memory store (replace with a real DB in production) ───────────────────
# Structure: { product_id: [{"id": str, "review": str, "sticker": str, "result": dict}] }
_PRODUCT_REVIEWS: dict[str, list[dict]] = defaultdict(list)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from e-commerce front-end


# ── Helper ────────────────────────────────────────────────────────────────────

def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok", "version": "2.0.0"})


@app.post("/analyze")
def analyze_single():
    """
    Body: { "review": "...", "sticker": "positive|negative|neutral" }
    """
    data = request.get_json(silent=True)
    if not data or "review" not in data:
        return _err("'review' field is required")
    result = analyze(
        text=data["review"],
        sticker=data.get("sticker", "neutral"),
    )
    return jsonify(result)


@app.post("/analyze/batch")
def analyze_batch_endpoint():
    """
    Body: { "reviews": ["...", "..."], "stickers": ["neutral", ...] }
    """
    data = request.get_json(silent=True)
    if not data or "reviews" not in data:
        return _err("'reviews' field is required")

    reviews  = data["reviews"]
    stickers = data.get("stickers")

    if not isinstance(reviews, list):
        return _err("'reviews' must be a list")

    results = analyze_batch(reviews, stickers=stickers)
    return jsonify({"results": results, "count": len(results)})


@app.get("/product/<product_id>")
def get_product_summary(product_id: str):
    """Return sentiment distribution for all reviews of a product."""
    entries  = _PRODUCT_REVIEWS.get(product_id, [])
    if not entries:
        return jsonify({
            "product_id"   : product_id,
            "total_reviews": 0,
            "counts"       : {},
            "percentages"  : {},
            "reviews"      : [],
        })

    reviews  = [e["review"]  for e in entries]
    stickers = [e["sticker"] for e in entries]
    summary  = product_dashboard(reviews, stickers=stickers)

    # Attach review-level detail
    review_list = []
    for i, entry in enumerate(entries):
        row = {
            "id"     : entry["id"],
            "review" : entry["review"],
            "sticker": entry["sticker"],
            "result" : entry["result"],
            "flagged": entry["result"]["label"] == "suspicious",
        }
        review_list.append(row)

    return jsonify({
        "product_id": product_id,
        **summary,
        "reviews"  : review_list,
    })


@app.post("/product/<product_id>/add")
def add_review(product_id: str):
    """
    Add a new review for a product.
    Body: { "review": "...", "sticker": "neutral" }
    """
    data = request.get_json(silent=True)
    if not data or "review" not in data:
        return _err("'review' field is required")

    review  = data["review"]
    sticker = data.get("sticker", "neutral")
    result  = analyze(review, sticker=sticker)

    entry = {
        "id"     : str(uuid.uuid4()),
        "review" : review,
        "sticker": sticker,
        "result" : result,
    }
    _PRODUCT_REVIEWS[product_id].append(entry)

    return jsonify({"message": "Review added", "entry": entry}), 201


@app.delete("/review/<review_id>")
def delete_review(review_id: str):
    """
    Remove a specific review (admin moderation action).
    Searches across all products.
    """
    for pid, entries in _PRODUCT_REVIEWS.items():
        for i, entry in enumerate(entries):
            if entry["id"] == review_id:
                removed = entries.pop(i)
                return jsonify({"message": "Review removed", "id": removed["id"]})
    return _err("Review not found", 404)


@app.get("/product/<product_id>/suspicious")
def get_suspicious(product_id: str):
    """Return only suspicious/negative reviews for admin review."""
    entries = _PRODUCT_REVIEWS.get(product_id, [])
    flagged = [
        e for e in entries
        if e["result"]["label"] in ("suspicious", "negative")
    ]
    return jsonify({"product_id": product_id, "flagged": flagged, "count": len(flagged)})


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
