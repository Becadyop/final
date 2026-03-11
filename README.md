# Echo Feeling

Echo Feeling is a Python package that analyzes customer emotions from text, emojis, and stickers.
It can be integrated into e-commerce websites, chat systems, or customer review analysis dashboards.

The package provides a simple sentiment engine and a REST API built with Flask so developers can easily connect their applications.

---

# Features

* Sentiment analysis for customer reviews
* REST API for easy integration
* Lightweight Python package
* Easy integration with e-commerce admin dashboards
* Works with any backend that can send HTTP requests

---

# Installation

Install the package using pip:

```
pip install echo-feeling
```

Or install from the wheel file:

```
pip install echo_feeling-1.0.0-py3-none-any.whl
```

---

# Quick Start

### Import the Engine

```python
from echo_feeling import EchoFeelingEngine

engine = EchoFeelingEngine()

result = engine.analyze(text="I love this product!")

print(result)
```

Example output:

```
{
 "sentiment": "neutral",
 "confidence": 0.9
}
```

---

# Running the API Server

You can start the REST API server like this:

```python
from echo_feeling.api import start_server

start_server()
```

The server will run at:

```
http://127.0.0.1:5050
```

---

# API Documentation

## Endpoint

```
POST /analyze
```

## Request Example

```
{
 "text": "Amazing product!"
}
```

## Response Example

```
{
 "sentiment": "neutral",
 "confidence": 0.9
}
```

---

# Python Integration Example

```python
import requests

response = requests.post(
    "http://localhost:5050/analyze",
    json={"text": "Great product!"}
)

print(response.json())
```

---

# JavaScript Integration Example

```javascript
fetch("http://localhost:5050/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Amazing product!" })
})
.then(res => res.json())
.then(data => console.log(data));
```

---

# Example Use Cases

* Customer review sentiment analysis
* E-commerce admin dashboards
* Customer feedback monitoring
* Chat emotion detection
* Product satisfaction analysis

---

# Project Structure

```
Echo_Feeling
│
├── echo_feeling
│   ├── __init__.py
│   ├── engine.py
│   └── api.py
│
├── deployment_phase
├── training_phase
│
├── setup.py
├── requirements.txt
└── README.md
```

---

# Future Improvements

* Deep learning sentiment model
* Emoji emotion detection
* Dashboard visualization
* Real-time review analytics
* SaaS deployment

---

# Author

Echo Feeling Project
Developed for sentiment analysis and emotion detection in e-commerce platforms.
