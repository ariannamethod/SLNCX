import sys


class _Tensor:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, _):
        return self

    def tolist(self):
        return self.data


class _TorchStub:
    long = int

    @staticmethod
    def tensor(data, dtype=None):
        return _Tensor(data)

    class no_grad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass


class _DummyEncodingStub:
    def encode(self, text):
        return [0]

    def decode(self, ids):
        return "dummy"


class _TiktokenStub:
    Encoding = object
    @staticmethod
    def get_encoding(name):
        return _DummyEncodingStub()


class _GPTStub:
    class GPTConfig:
        pass

    class GPT:
        pass


sys.modules.setdefault("torch", _TorchStub())
sys.modules.setdefault("tiktoken", _TiktokenStub())
sys.modules.setdefault("nanogpt_model", _GPTStub)

import torch  # noqa: E402
import wulf_inference  # noqa: E402


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
    monkeypatch.setattr(wulf_inference, "ENCODER", DummyEncoding())

    result = wulf_inference.generate("hi")
    assert result == "dummy"
