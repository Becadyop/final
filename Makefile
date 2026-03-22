# Echo Feeling – Makefile
# Usage: make <target>

.PHONY: install data train serve node-serve test clean

## Install Python dependencies
install:
	pip install -e ".[server,dev]"

## Generate sample dataset (3000 reviews + sticker folders)
data:
	python generate_sample_data.py --out_dir data --n 3000

## Train all models (run `make data` first)
train:
	python training_phase/train.py \
		--data_path data/reviews.csv \
		--sticker_root data/stickers \
		--output_dir models

## Train with BERT embeddings (slow — requires transformers + torch)
train-bert:
	python training_phase/train.py \
		--data_path data/reviews.csv \
		--sticker_root data/stickers \
		--output_dir models \
		--use_bert

## Start the Flask inference API server
serve:
	python deployment_phase/server.py --port 5000

## Start the Node.js Express server (Flask must be running)
node-serve:
	node deployment_phase/app.js

## Run unit tests
test:
	pytest tests/ -v --tb=short

## Full pipeline: install → data → train → serve
all: install data train serve

## Remove generated files
clean:
	rm -rf models/ data/ __pycache__/ .pytest_cache/ *.egg-info/
	find . -name "*.pyc" -delete
