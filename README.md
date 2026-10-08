# RAG on AWS: from prototype to production

A small, production-minded retrieval-augmented generation service. It answers questions over a document set with citations, refuses when the documents don't contain the answer, reports tokens, latency and cost per request, and ships with an evaluation harness and Terraform for AWS (Bedrock + Lambda + API Gateway).

The sample documents describe **Ridgeline Gear**, a fictional outdoor store.

## What this demonstrates

| Concern | How it's handled |
|---|---|
| Grounded answers | System prompt restricts the model to numbered sources and requires citations. Source text is treated as data, not instructions. |
| Knowing when not to answer | A similarity cutoff (`RAG_MIN_SCORE`) skips the model call entirely when nothing relevant is retrieved. |
| Measurable retrieval quality | `rag-eval` reports hit rate@k, MRR and refusal rate on unanswerable questions from a labeled set. |
| Cost visibility | Every request logs input/output tokens and cost (from prices you configure; nothing is guessed). |
| Observability | One structured JSON log line per request, ready for CloudWatch Logs Insights. Question text is never logged. |
| Cost guardrails in AWS | API Gateway throttling plus an AWS Budgets alert. |
| Least privilege | The Lambda role can read only the index prefix in S3 and invoke Bedrock models. |
| Local development | Same code runs against Ollama (free) or a deterministic fake provider (tests, CI). |

## Architecture

```
            ┌──────────────┐      ┌───────────────────────────┐
 client ──► │ API Gateway  │ ───► │ Lambda (container image)  │
            │ HTTP API     │      │ FastAPI + Mangum          │
            │ (throttled)  │      │                           │
            └──────────────┘      │ 1. embed question ────────┼──► Bedrock (Titan embeddings)
                                  │ 2. cosine search in index │
                                  │ 3. cutoff → refuse        │
                                  │ 4. answer w/ citations ───┼──► Bedrock (chat model, Converse API)
                                  └─────────────┬─────────────┘
                                                │ cold start: download index
                                          ┌─────▼─────┐
                                          │    S3     │  chunks.json + vectors.npy
                                          └───────────┘
```

Ingestion runs offline (`rag-ingest`) and uploads the index to S3. A flat numpy index fits a few thousand chunks and keeps the Lambda small; beyond that, the `store` module is the single place to swap in pgvector or OpenSearch.

## Results

Retrieval evaluation on `evals/questions.jsonl`, `top_k=4`. 15 answerable questions (including two whose answer is an implied "no", like "Do you ship to Brazil?") and 7 the documents can't answer: 2 off-topic and 5 near-misses in the store's own domain (opening hours, PayPal, kayaks, student discount, changing the account email).

| Embeddings | `min_score` | hit rate@4 | MRR | refusal on unanswerable |
|---|---|---|---|---|
| Fake (hashed bag-of-words baseline) | 0.10 | 1.00 | 0.97 | 0.43 |
| `nomic-embed-text` (Ollama) | 0.40 | 1.00 | 1.00 | 0.00 |
| `nomic-embed-text` (Ollama) | 0.50 | 1.00 | 1.00 | 0.29 |
| `nomic-embed-text` (Ollama) | 0.55 | 1.00 | 1.00 | 0.86 |
| `nomic-embed-text` (Ollama) | 0.65 | 1.00 | 1.00 | 1.00 |
| `nomic-embed-text` (Ollama) | 0.70 | 0.80 | 0.80 | 1.00 |

What this shows:
- The cutoff is model-specific and the usable band is narrow: 0.65 is clean, 0.70 already drops real questions ("Do you ship to Canada?"). Near-miss questions are what make the threshold hard; off-topic ones are easy.
- The threshold was tuned on this same set, so treat 0.65 as a starting point. A real deployment needs a held-out set built from actual user questions.
- A similarity cutoff can't handle "no by omission" (Brazil isn't in the shipping list). Those must reach the model, which is why the system prompt allows "I don't know" and generation needs its own evals.

Re-run `rag-eval` whenever chunking, embeddings or `top_k` change.

**Known limitation:** this harness measures retrieval only. In a local run with a small code model (`qwen2.5-coder:7b`), retrieval found the right section for "Can I return a tent I already used?" but the model still answered it wrong. Generation quality needs its own evaluation (expected-answer checks or an LLM judge with a rubric). That's the next step on the roadmap below.

## Verification status

- Verified: offline test suite (including Bedrock and S3 adapters against stubbed clients), `terraform validate`, the Lambda container invoked locally through the Lambda runtime emulator, and the eval above with local Ollama models.
- Not yet verified: a real deploy, real Bedrock and S3 calls, and the IAM policy in a live account.

## Run locally

```bash
uv venv && uv pip install -e '.[dev,aws]'

# offline, deterministic
RAG_PROVIDER=fake RAG_MIN_SCORE=0.1 rag-ingest data/docs
RAG_PROVIDER=fake RAG_MIN_SCORE=0.1 rag-eval evals/questions.jsonl
pytest

# real embeddings and a local model
ollama pull nomic-embed-text && ollama pull <chat model>
export RAG_PROVIDER=ollama OLLAMA_CHAT_MODEL=<chat model> RAG_MIN_SCORE=0.65
rag-ingest data/docs && uvicorn rag_demo.api:app --reload
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"Do you ship to Canada?"}'
```

## Deploy to AWS

Prerequisites: AWS CLI credentials, Docker, Terraform ≥ 1.6, at least one git commit (the image tag is the commit sha), the `aws` extra installed (`uv pip install -e '.[aws]'`), and access enabled in the Bedrock console for the embedding model and the chat model you pick.

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # fill in chat_model, alert_email, prices

# 1. registry and bucket first (the Lambda needs an image to exist)
terraform init
terraform apply -target=aws_ecr_repository.app -target=aws_s3_bucket.index -var image_tag=bootstrap

# 2. build and push the image (immutable tags: use the git sha)
REPO=$(terraform output -raw ecr_repository_url); TAG=$(git rev-parse --short HEAD)
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%%/*}"
docker build -t "$REPO:$TAG" ../.. && docker push "$REPO:$TAG"

# 3. build the index with Bedrock embeddings and upload it
(cd ../.. && RAG_PROVIDER=bedrock RAG_INDEX_PATH=/tmp/index rag-ingest data/docs)
aws s3 cp /tmp/index "$(terraform output -raw index_s3_uri)" --recursive

# 4. everything else
terraform apply -var image_tag=$TAG
curl -s "$(terraform output -raw api_url)ask" -H 'content-type: application/json' -d '{"question":"Do you ship to Canada?"}'
```

Re-tune `min_score` with `rag-eval` against Bedrock embeddings before relying on refusals (set `min_score` in `terraform.tfvars`); the 0.65 above is for `nomic-embed-text`.

The budget alert only emails; it does not stop spend. API Gateway throttling is the hard limit on request rate.

Tear down with `terraform destroy`.

## Roadmap

- Generation evals: expected-answer assertions plus an LLM judge for faithfulness to sources
- Hybrid retrieval (BM25 + vectors) and a reranker
- Incremental ingestion triggered by S3 uploads
- Auth on the API (Cognito or IAM) for anything beyond a demo

## Layout

```
rag_demo/      config, providers (bedrock / ollama / fake), chunking + index, RAG, API, eval
data/docs/     fictional sample documents
evals/         labeled questions
infra/         Terraform for AWS
tests/         pytest, runs offline
```
