import random
import string
import sys
import unittest
from pathlib import Path
from unittest import mock

from autogpt.config import Config
from autogpt.memory.local import LocalCache


class TestLocalCache(unittest.TestCase):
    def random_string(self, length):
        return "".join(random.choice(string.ascii_letters) for _ in range(length))

    def setUp(self):
        cfg = cfg = Config()
        cfg.memory_index = "test-cache"
        embeddings = {}

        def fake_embed(text):
            if text not in embeddings:
                vec = [0.0] * 1536
                index = len(embeddings)
                vec[index % 1536] = 1.0
                embeddings[text] = vec
            return embeddings[text]

        patch_base = mock.patch(
            "autogpt.memory.base.get_ada_embedding", side_effect=fake_embed
        )
        patch_local = mock.patch(
            "autogpt.memory.local.get_ada_embedding", side_effect=fake_embed
        )
        patch_base.start()
        patch_local.start()
        self.addCleanup(patch_base.stop)
        self.addCleanup(patch_local.stop)

        self.cache = LocalCache(cfg)
        self.cache.clear()

        # Add example texts to the cache
        self.example_texts = [
            "The quick brown fox jumps over the lazy dog",
            "I love machine learning and natural language processing",
            "The cake is a lie, but the pie is always true",
            "ChatGPT is an advanced AI model for conversation",
        ]

        for text in self.example_texts:
            self.cache.add(text)

        # Add some random strings to test noise
        for _ in range(5):
            self.cache.add(self.random_string(10))

    def test_get_relevant(self):
        query = "I'm interested in artificial intelligence and NLP"
        k = 3
        relevant_texts = self.cache.get_relevant(query, k)

        print(f"Top {k} relevant texts for the query '{query}':")
        for i, text in enumerate(relevant_texts, start=1):
            print(f"{i}. {text}")

        self.assertEqual(len(relevant_texts), k)
        self.assertIn(self.example_texts[1], relevant_texts)


if __name__ == "__main__":
    unittest.main()
