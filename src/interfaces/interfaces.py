"""Модуль для задания базовых интерфейсов"""
from abc import ABC, abstractmethod
from typing import Any
from numpy import ndarray
from torch import Tensor


class IStorage(ABC):
    """Абстрактный класс векторного хранилища"""
    @abstractmethod
    def get_data(self, query_embedding: Tensor | ndarray | list[Tensor],
                 top_k: int) -> list[Any]:
        """
        Получить данные из хранилища
        :param query_embedding: векторное представление запроса
        :param top_k: количество лучших чанков
        :return:
        """

    @abstractmethod
    def save_data(self, embeddings: Tensor | ndarray | list[Tensor],
                  chunks: list[str], metadata: dict):
        """
        Сохранить вектора в хранилище
        :param embeddings: векторное представление чанков
        :param chunks: чанки текста
        :param metadata: метаданные чанков
        :return:
        """
    @abstractmethod
    def delete_by_page_id(self, page_id: int):
        """
        Удалить все чанки определенного файла
        :file_name: имя файла
        :return:
        """


class IRetriever(ABC):
    """Абстрактный класс для Ретривера"""
    @abstractmethod
    def generate_embeddings(self, chunks: list[str], metadata: list[dict]):
        """
        Сгенерировать вектора из кусков текста (стоит добавить опцию нормализацию векторов?)
        :param chunks:
        :param metadata:
        :return:
        """
    @abstractmethod
    def find_similar_context(self, query: str) -> list[(str, str)]:
        """
        Парсит в вектора query, затем ищет в storage, ранжирует и отдает возможный контекст
        :param query:
        :return:
        """
    @abstractmethod
    def best_match (self, query_list: list[str], context_list: list[(str, str)], top_k: int) -> str:
        """
        :param query_list:
        :param context_list:
        :param top_k:
        :return:
        """


class IGenerator(ABC):
    """Абстрактный класс Генератора"""
    @abstractmethod
    def generate_answer(self, query: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        """
        Генерирует ответ на запрос с учётом контекста, полученного от ретривера.

        :param query: Запрос, для которого нужно сгенерировать ответ
        :param temperature: Температура для генерации (по умолчанию 0.2)
        :param max_tokens: Максимальное количество токенов для генерации (по умолчанию 600)
        :return: Сгенерированный ответ
        """
