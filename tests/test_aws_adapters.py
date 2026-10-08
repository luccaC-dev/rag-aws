"""Bedrock and S3 adapters against stubbed clients: checks request shapes and response parsing
without AWS credentials or network."""

import io
import json

import numpy as np

from rag_demo import api
from rag_demo.providers import BedrockChat, BedrockEmbedder


class StubBedrock:
    def __init__(self):
        self.calls = []

    def invoke_model(self, modelId, body):
        self.calls.append(("invoke_model", modelId, json.loads(body)))
        return {"body": io.BytesIO(json.dumps({"embedding": [3.0, 4.0]}).encode())}

    def converse(self, **kwargs):
        self.calls.append(("converse", kwargs))
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "Yes [1]"}, {"text": "."}]}},
            "usage": {"inputTokens": 120, "outputTokens": 7, "totalTokens": 127},
        }


def test_bedrock_embedder_sends_titan_payload_and_normalizes():
    emb = BedrockEmbedder.__new__(BedrockEmbedder)
    emb.client, emb.model = StubBedrock(), "amazon.titan-embed-text-v2:0"
    vectors = emb.embed(["a", "b"])
    assert [c[2] for c in emb.client.calls] == [{"inputText": "a"}, {"inputText": "b"}]
    assert np.allclose(vectors, [[0.6, 0.8], [0.6, 0.8]])


def test_bedrock_chat_uses_converse_and_reads_usage():
    chat = BedrockChat.__new__(BedrockChat)
    chat.client, chat.model = StubBedrock(), "some-model-id"
    result = chat.chat("system prompt", "user prompt")
    _, kwargs = chat.client.calls[0]
    assert kwargs["modelId"] == "some-model-id"
    assert kwargs["system"] == [{"text": "system prompt"}]
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "user prompt"}]}]
    assert result.text == "Yes [1]."
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 7)


def test_bedrock_chat_requires_model_id():
    try:
        BedrockChat("us-east-1", "")
    except ValueError as e:
        assert "BEDROCK_CHAT_MODEL" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_index_download_builds_s3_keys(tmp_path, monkeypatch):
    downloads = []

    class StubS3:
        def download_file(self, bucket, key, dest):
            downloads.append((bucket, key, dest))

    import boto3

    monkeypatch.setattr(boto3, "client", lambda name: StubS3())
    api._fetch_index_from_s3("s3://my-bucket/index/", str(tmp_path))
    assert downloads == [
        ("my-bucket", "index/chunks.json", str(tmp_path / "chunks.json")),
        ("my-bucket", "index/vectors.npy", str(tmp_path / "vectors.npy")),
    ]
