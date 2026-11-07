# PubMed Abstract Summarizer (Local Llama 3.3 70B via vLLM)

Cleans messy PubMed-style JSON and asks a locally hosted Llama 3.3 70B
(vLLM, OpenAI-compatible API) to summarize each study's clinical outcome in one sentence.

## Features
- Handles inconsistent key names (`ArticleTitle` / `title`, `abstract_text` / `Abstract`, ...)
- Handles null values, placeholder strings ("Unknown") and corrupted entries (`ERR_*` IDs)
- Skips records with no abstract instead of wasting an LLM call
- Timeouts, connection drops and 429/5xx errors are retried with exponential backoff + jitter
- A failed record never crashes the batch; its status is saved in the output file

## Setup
```bash
pip install -r requirements.txt
python pubmed_summarizer.py --dry-run        # clean data + show payloads, no server needed
python pubmed_summarizer.py                  # send to the server, writes summaries.json
python pubmed_summarizer.py --input my.json  # use your own data file
```
Edit `API_URL` and `MODEL_NAME` at the top of the script if your server differs.

## Timeout behaviour
Connect timeout is 5s; read timeout is 120s. A server that takes 45s to respond
succeeds normally. If a request does time out, it is retried up to 3 times
(2s, 4s backoff + jitter), then that record is marked as an error and processing continues.

## Server: avoiding OOM on DGX Spark
Llama 3.3 70B in BF16 needs ~140 GB, more than the Spark's 128 GB unified memory,
so use an FP8 (or 4-bit) checkpoint. Replace `<TAG>` with a current NGC vLLM tag
for aarch64/Blackwell.

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

docker run -d --name vllm-llama70b \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:<TAG>-py3 \
  vllm serve nvidia/Llama-3.3-70B-Instruct-FP8 \
    --served-model-name llama-3.3-70b \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --kv-cache-dtype fp8
```

| Argument | Why |
|---|---|
| FP8 checkpoint | ~70 GB weights instead of ~140 GB |
| `--gpu-memory-utilization 0.80` | unified memory is shared with the OS; leave headroom |
| `--max-model-len 8192` | abstracts are short; avoids reserving KV cache for 128k context |
| `--max-num-seqs 4` | caps concurrent requests and KV cache growth |
| `--kv-cache-dtype fp8` | halves KV cache memory |
| `--served-model-name` | must match `MODEL_NAME` in the script |
| `--host 0.0.0.0` | makes the API reachable on the LAN |
