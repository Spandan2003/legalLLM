#!/bin/bash
docker stop hallucination-spandan
docker rm hallucination-spandan

set -a
source .env
set +a

model=meta-llama/Meta-Llama-3.1-8B-Instruct
volume=./cache/HF_CACHE # share a volume with the Docker container to avoid downloading weights every run

docker run -d --name hallucination-spandan --gpus '"device=1"' --shm-size 40g -p 8080:80 -e HF_TOKEN=$HF_TOKEN -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.1.0 \
    --model-id $model \
    --max-best-of 5