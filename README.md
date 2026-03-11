# Echo Feeling

Echo Feeling is a multimodal emotion analysis engine designed for ecommerce interaction analysis.

It analyzes:

- Text sentiment
- Emoji sentiment
- Sticker sentiment

## Installation

Clone the repository:

git clone https://github.com/Phantom-rv7/Echo_Feeling.git

Install dependencies:

pip install -r requirements.txt

## Run the API

python -m echo_feeling.api

Server will start at:

http://localhost:5050

## API Endpoint

POST /analyze

Example request:

{
"text": "I love this product",
"emoji": "😍"
}