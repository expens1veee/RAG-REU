import unittest
from unittest.mock import AsyncMock, MagicMock
import pytest
import numpy as np
from src.internal.retriever.retriever import Retriever
from src.interfaces.interfaces import IStorage


@pytest.mark.asyncio
class TestRetriever(unittest.TestCase):
    def setUp(self):
        # Создаём mock для IStorage
        self.storage_mock = MagicMock(spec=IStorage)
        self.storage_mock.save_data = AsyncMock()
        self.storage_mock.get_data = AsyncMock()

        # Инициализируем Retriever с mock-объектом storage
        self.retriever = Retriever(storage=self.storage_mock, model_name="intfloat/multilingual-e5-large")

    async def test_generate_embeddings(self):
        # Тестируем метод generate_embeddings
        chunks = ["The capital of France is Paris.", "France is in Europe."]
        metadata = [{"page_id": 1, "source": "doc1"}, {"page_id": 1, "source": "doc1"}]

        await self.retriever.generate_embeddings(chunks, metadata)

        # Проверяем, что save_data вызван
        self.storage_mock.save_data.assert_called_once()
        args, kwargs = self.storage_mock.save_data.call_args
        embeddings, saved_chunks, saved_metadata = args

        # Проверяем, что эмбеддинги сгенерированы
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 768)
        self.assertEqual(saved_chunks, chunks)
        self.assertEqual(saved_metadata, metadata[0])

    async def test_generate_embeddings_mismatch(self):
        # Тестируем ошибку при несовпадении количества чанков и метаданных
        chunks = ["Chunk 1"]
        metadata = []

        with self.assertRaises(ValueError):
            await self.retriever.generate_embeddings(chunks, metadata)

    async def test_find_similar_context(self):
        # Тестируем метод find_similar_context
        query = "What is the capital of France?"

        # Настраиваем mock для storage.get_data
        mock_result = [
            {"text": "The capital of France is Paris.", "metadata": {"page_id": 1, "source": "doc1"}, "score": 0.95},
            {"text": "France is in Europe.", "metadata": {"page_id": 1, "source": "doc1"}, "score": 0.85}
        ]
        self.storage_mock.get_data.return_value = mock_result

        results = await self.retriever.find_similar_context(query)

        # Проверяем результат
        expected = [
            ("The capital of France is Paris.", "doc1"),
            ("France is in Europe.", "doc1")
        ]
        self.assertEqual(results, expected)
        self.storage_mock.get_data.assert_called_once()

    async def test_best_match(self):
        # Тестируем метод best_match
        query_list = ["What is the capital of France?", "Where is France located?"]
        context_list = [
            ("The capital of France is Paris.", "doc1"),
            ("France is in Europe.", "doc1")
        ]
        top_k = 1

        best_chunk = await self.retriever.best_match(query_list, context_list, top_k)

        # Проверяем, что возвращён наиболее релевантный чанк
        self.assertIn(best_chunk, [context[0] for context in context_list])

    async def test_best_match_empty_input(self):
        # Тестируем ошибку при пустых входных данных
        with self.assertRaises(ValueError):
            await self.retriever.best_match([], [], 1)


if __name__ == "__main__":
    unittest.main()
