import torch
import wulf_inference


def test_generate(monkeypatch):
    class DummyModel:
        def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):
            return torch.tensor([[1, 2, 3]])

    class DummyEncoding:
        def encode(self, text):
            return [0]

        def decode(self, ids):
            return "dummy"

    def dummy_load_model():
        return DummyModel()

    monkeypatch.setattr(wulf_inference, "load_model", dummy_load_model)
    monkeypatch.setattr(wulf_inference.tiktoken, "get_encoding", lambda name: DummyEncoding())

    result = wulf_inference.generate("hi")
    assert result == "dummy"
