.PHONY: setup train eval app docker-build docker-run

setup:
	python -m pip install -r requirements.txt

train:
	python scripts/train.py --config configs/config.yaml

eval:
	python scripts/eval.py --weights runs/ckpt.pt --data-root data --img-size 512

app:
	streamlit run app.py --server.port=8501 --server.address=0.0.0.0

docker-build:
	docker build -t $(shell basename $(PWD)):latest .

docker-run:
	docker run --rm -it -p 8501:8501 -v $(PWD)/data:/app/data -v $(PWD)/runs:/app/runs $(shell basename $(PWD)):latest
