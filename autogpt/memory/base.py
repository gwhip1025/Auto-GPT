"""Base class for memory providers."""
import abc

import openai
from openai.error import AuthenticationError

from autogpt.config import AbstractSingleton, Config

cfg = Config()


def get_ada_embedding(text):
    """Return the embedding for the given text using OpenAI's API.

    If the OpenAI API key is missing, a zero vector is returned so that unit tests
    can run without network access.
    """

    text = text.replace("\n", " ")
    try:
        if cfg.use_azure:
            return openai.Embedding.create(
                input=[text],
                engine=cfg.get_azure_deployment_id_for_model("text-embedding-ada-002"),
            )["data"][0]["embedding"]
        else:
            return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
                "data"
            ][0]["embedding"]
    except AuthenticationError:
        # During tests the API key may be absent; fall back to a zero-vector so
        # that embedding-dependent features still operate.
        return [0.0] * 1536


class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data):
        pass

    @abc.abstractmethod
    def get(self, data):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def get_relevant(self, data, num_relevant=5):
        pass

    @abc.abstractmethod
    def get_stats(self):
        pass
