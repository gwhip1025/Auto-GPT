import os
import sys
import unittest
from unittest import mock

from autogpt.memory.local import LocalCache, CacheContent


def MockConfig():
    return type(
        "MockConfig",
        (object,),
        {
            "debug_mode": False,
            "continuous_mode": False,
            "speak_mode": False,
            "memory_index": "auto-gpt",
        },
    )


class TestLocalCache(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        # Ensure each test starts with a clean cache file
        if os.path.exists(f"{self.cfg.memory_index}.json"):
            os.remove(f"{self.cfg.memory_index}.json")
        # Reset singleton instances to avoid cross-test pollution
        import autogpt.config
        autogpt.config.Singleton._instances = {}
        # Patch embeddings to deterministic vectors to avoid OpenAI dependency
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
        self.addCleanup(patch_base.stop)
        self.addCleanup(patch_local.stop)
        patch_base.start()
        patch_local.start()

        self.cache = LocalCache(self.cfg)

    def test_add(self):
        text = "Sample text"
        self.cache.add(text)
        self.assertIn(text, self.cache.data.texts)

    def test_clear(self):
        self.cache.clear()
        self.assertEqual(self.cache.data.texts, [])
        self.assertEqual(
            tuple(self.cache.data.embeddings.shape), (0, 1536)
        )

    def test_get(self):
        text = "Sample text"
        self.cache.add(text)
        result = self.cache.get(text)
        self.assertEqual(result, [text])

    def test_get_relevant(self):
        text1 = "Sample text 1"
        text2 = "Sample text 2"
        self.cache.add(text1)
        self.cache.add(text2)
        result = self.cache.get_relevant(text1, 1)
        self.assertEqual(result, [text1])

    def test_get_stats(self):
        text = "Sample text"
        self.cache.add(text)
        stats = self.cache.get_stats()
        self.assertEqual(stats[0], 1)
        self.assertEqual(stats[1], self.cache.data.embeddings.shape)


if __name__ == "__main__":
    unittest.main()
