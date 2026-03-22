/**
 * Echo Feeling – Express.js Integration Example
 * ================================================
 * Shows how to wire the sentiment_module into a Node.js e-commerce back-end.
 *
 * Install dependencies:
 *   npm install express
 *
 * Start (make sure Flask server is running on port 5000 first):
 *   node app.js
 */

const express  = require("express");
const path     = require("path");
const sentiment = require("./sentiment_module");

const app  = express();
const PORT = process.env.PORT || 3000;

// ── Middleware ────────────────────────────────────────────────────────────
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Attach sentiment helpers to every request
app.use(sentiment.sentimentMiddleware());

// ── Routes ────────────────────────────────────────────────────────────────

/**
 * POST /api/products/:id/reviews
 * Add a review for a product and return the sentiment result.
 */
app.post("/api/products/:id/reviews", async (req, res) => {
  const { text, sticker = "neutral" } = req.body;
  if (!text) return res.status(400).json({ error: "text is required" });

  try {
    const result = await req.sentiment.addReview(req.params.id, text, sticker);
    res.status(201).json(result);
  } catch (err) {
    res.status(503).json({ error: "Sentiment service unavailable", detail: err.message });
  }
});

/**
 * GET /api/products/:id/dashboard
 * Admin panel dashboard data for a product.
 */
app.get("/api/products/:id/dashboard", async (req, res) => {
  try {
    const summary = await req.sentiment.getProductSummary(req.params.id);
    res.json(summary);
  } catch (err) {
    res.status(503).json({ error: "Sentiment service unavailable", detail: err.message });
  }
});

/**
 * GET /api/products/:id/suspicious
 * Returns reviews flagged as suspicious or negative.
 */
app.get("/api/products/:id/suspicious", async (req, res) => {
  try {
    const result = await req.sentiment.getSuspiciousReviews(req.params.id);
    res.json(result);
  } catch (err) {
    res.status(503).json({ error: "Sentiment service unavailable", detail: err.message });
  }
});

/**
 * DELETE /api/reviews/:id
 * Admin moderation: remove a review by ID.
 */
app.delete("/api/reviews/:id", async (req, res) => {
  try {
    const result = await req.sentiment.deleteReview(req.params.id);
    res.json(result);
  } catch (err) {
    res.status(503).json({ error: "Sentiment service unavailable", detail: err.message });
  }
});

/**
 * POST /api/analyze
 * Quick one-off sentiment check (for live preview in review form).
 */
app.post("/api/analyze", async (req, res) => {
  const { text, sticker = "neutral" } = req.body;
  if (!text) return res.status(400).json({ error: "text is required" });
  try {
    const result = await req.sentiment.analyze(text, sticker);
    res.json(result);
  } catch (err) {
    res.status(503).json({ error: "Sentiment service unavailable", detail: err.message });
  }
});

/**
 * GET /admin
 * Serve the admin dashboard HTML.
 */
app.get("/admin", (_req, res) => {
  res.sendFile(path.join(__dirname, "admin_dashboard.html"));
});

/**
 * GET /health
 * Health-check for this Node server + underlying sentiment API.
 */
app.get("/health", async (_req, res) => {
  const apiUp = await sentiment.healthCheck();
  res.json({ node: "ok", sentiment_api: apiUp ? "ok" : "down" });
});

// ── Start ─────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🛍️  Echo Feeling e-commerce server running at http://localhost:${PORT}`);
  console.log(`📊  Admin dashboard: http://localhost:${PORT}/admin`);
  console.log(`🔌  Expecting Flask sentiment API at: ${process.env.ECHO_FEELING_API_URL || "http://localhost:5000"}\n`);
});

module.exports = app;
