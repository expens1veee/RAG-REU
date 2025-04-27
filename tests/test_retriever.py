import pytest
from unittest.mock import MagicMock
from src.internal.retriever.retriever import Retriever


@pytest.fixture
def mock_storage():
    return MagicMock()


@pytest.fixture
def retriever(mock_storage):
    return Retriever(storage=mock_storage)


def test_generate_embeddings(retriever, mock_storage):
    chunks = ["Пример текста"]
    metadata = [{"source": "тест"}]

    result = retriever.generate_embeddings(chunks, metadata)

    # Проверка, что метод save_data вызван корректно
    mock_storage.save_data.assert_called_once()
    assert result.shape[0] == len(chunks)


def test_generate_embeddings_empty_chunks(retriever, mock_storage):
    chunks = []
    metadata = []

    retriever.generate_embeddings(chunks, metadata)

    # Проверяем, что save_data не вызывается
    mock_storage.save_data.assert_not_called()


def test_find_similar_context(retriever, mock_storage):
    mock_storage.get_data.return_value = [
        {"text": "Это тестовый текст", "metadata": {"source": "файл1"}},
        {"text": "Другой текст", "metadata": {"source": "файл2"}},
    ]

    result = retriever.find_similar_context("тестовый запрос")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == ("Это тестовый текст", "файл1")


def test_best_match(retriever):
    query_list = ["что делает кошка?"]
    context_list = [
        ("Кошка сидит на дереве", "source1"),
        ("Собака лает на прохожего", "source2"),
        ("Погода сегодня хорошая", "source3"),
    ]

    result = retriever.best_match(query_list, context_list, top_k=1)

    assert isinstance(result, str)
    assert result in [ctx[0] for ctx in context_list]


def test_best_match_with_empty_context(retriever):
    query_list = ["что делает кошка?"]
    context_list = []

    result = retriever.best_match(query_list, context_list, top_k=1)

    assert result == ""
