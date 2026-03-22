/**
 * Echo Feeling – Node.js Sentiment Module
 * ========================================
 * Lightweight wrapper around the Python Flask inference server.
 * Drop this file into any Node.js / Express e-commerce back-end.
 *
 * Usage (CommonJS):
 *   const sentiment = require('./sentiment_module');
 *   const result = await sentiment.analyze("Great product! 😍");
 *   console.log(result.label); // "positive"
 *
 * Usage (ESM):
 *   import * as sentiment from './sentiment_module.js';
 */

const http  = require("http");
const https = require("https");
const url   = require("url");

// ── Configuration ────────────────────────────────────────────────────────────
const DEFAULT_BASE_URL = process.env.ECHO_FEELING_API_URL || "http://localhost:5000";
const DEFAULT_TIMEOUT  = parseInt(process.env.ECHO_FEELING_TIMEOUT || "8000", 10);

// ── HTTP helper ──────────────────────────────────────────────────────────────

function _request(method, endpoint, body = null, baseUrl = DEFAULT_BASE_URL) {
  return new Promise((resolve, reject) => {
    const parsed  = url.parse(`${baseUrl}${endpoint}`);
    const lib     = parsed.protocol === "https:" ? https : http;
    const payload = body ? JSON.stringify(body) : null;

    const options = {
      hostname: parsed.hostname,
      port    : parsed.port || (parsed.protocol === "https:" ? 443 : 80),
      path    : parsed.path,
      method,
      headers : {
        "Content-Type": "application/json",
        ...(payload ? { "Content-Length": Buffer.byteLength(payload) } : {}),
      },
    };

    const req = lib.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          resolve(JSON.parse(data));
        } catch {
          reject(new Error(`Invalid JSON response: ${data}`));
        }
      });
    });

    req.setTimeout(DEFAULT_TIMEOUT, () => {
      req.destroy(new Error(`Request timed out after ${DEFAULT_TIMEOUT}ms`));
    });

    req.on("error", reject);
    if (payload) req.write(payload);
    req.end();
  });
}

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Analyse a single review.
 *
 * @param {string} text          - Raw review text (may include emojis)
 * @param {string} [sticker]     - Sticker sentiment label: "positive" | "negative" | "neutral"
 * @param {string} [baseUrl]     - Override API base URL
 * @returns {Promise<{label: string, confidence: number, scores: object, emoji_score: number}>}
 */
async function analyze(text, sticker = "neutral", baseUrl = DEFAULT_BASE_URL) {
  if (!text || typeof text !== "string") throw new Error("text must be a non-empty string");
  return _request("POST", "/analyze", { review: text, sticker }, baseUrl);
}

/**
 * Analyse a batch of reviews.
 *
 * @param {string[]} texts      - Array of review strings
 * @param {string[]} [stickers] - Optional array of sticker labels
 * @param {string}   [baseUrl]
 * @returns {Promise<{results: object[], count: number}>}
 */
async function analyzeBatch(texts, stickers = null, baseUrl = DEFAULT_BASE_URL) {
  if (!Array.isArray(texts)) throw new Error("texts must be an array");
  const body = { reviews: texts };
  if (stickers) body.stickers = stickers;
  return _request("POST", "/analyze/batch", body, baseUrl);
}

/**
 * Get full sentiment summary for a product (for admin panel dashboard).
 *
 * @param {string} productId
 * @param {string} [baseUrl]
 * @returns {Promise<object>}  Includes counts, percentages, flagged reviews, etc.
 */
async function getProductSummary(productId, baseUrl = DEFAULT_BASE_URL) {
  return _request("GET", `/product/${encodeURIComponent(productId)}`, null, baseUrl);
}

/**
 * Add a new review to a product and get its sentiment result.
 *
 * @param {string} productId
 * @param {string} text
 * @param {string} [sticker]
 * @param {string} [baseUrl]
 * @returns {Promise<{message: string, entry: object}>}
 */
async function addReview(productId, text, sticker = "neutral", baseUrl = DEFAULT_BASE_URL) {
  return _request(
    "POST",
    `/product/${encodeURIComponent(productId)}/add`,
    { review: text, sticker },
    baseUrl,
  );
}

/**
 * Delete / moderate a review by its ID (admin action).
 *
 * @param {string} reviewId
 * @param {string} [baseUrl]
 * @returns {Promise<{message: string, id: string}>}
 */
async function deleteReview(reviewId, baseUrl = DEFAULT_BASE_URL) {
  return _request("DELETE", `/review/${encodeURIComponent(reviewId)}`, null, baseUrl);
}

/**
 * Get suspicious / negative reviews for admin review for a product.
 *
 * @param {string} productId
 * @param {string} [baseUrl]
 * @returns {Promise<{product_id: string, flagged: object[], count: number}>}
 */
async function getSuspiciousReviews(productId, baseUrl = DEFAULT_BASE_URL) {
  return _request("GET", `/product/${encodeURIComponent(productId)}/suspicious`, null, baseUrl);
}

/**
 * Check if the sentiment API server is running.
 *
 * @param {string} [baseUrl]
 * @returns {Promise<boolean>}
 */
async function healthCheck(baseUrl = DEFAULT_BASE_URL) {
  try {
    const res = await _request("GET", "/health", null, baseUrl);
    return res.status === "ok";
  } catch {
    return false;
  }
}

// ── Express middleware (optional) ────────────────────────────────────────────

/**
 * Express middleware: attaches `req.sentiment` helper to every request.
 * Usage: app.use(sentimentMiddleware());
 */
function sentimentMiddleware(baseUrl = DEFAULT_BASE_URL) {
  return (_req, _res, next) => {
    _req.sentiment = {
      analyze        : (text, sticker)            => analyze(text, sticker, baseUrl),
      analyzeBatch   : (texts, stickers)           => analyzeBatch(texts, stickers, baseUrl),
      getProductSummary : (pid)                    => getProductSummary(pid, baseUrl),
      addReview      : (pid, text, sticker)        => addReview(pid, text, sticker, baseUrl),
      deleteReview   : (rid)                       => deleteReview(rid, baseUrl),
      getSuspiciousReviews: (pid)                  => getSuspiciousReviews(pid, baseUrl),
    };
    next();
  };
}

// ── Exports ───────────────────────────────────────────────────────────────────
module.exports = {
  analyze,
  analyzeBatch,
  getProductSummary,
  addReview,
  deleteReview,
  getSuspiciousReviews,
  healthCheck,
  sentimentMiddleware,
};
