from src.interfaces.interfaces import IGenerator
from src.internal.retriever.retriever import Retriever
from src.interfaces.interfaces import IRetriever
from openai import OpenAI
from typing import List, Optional


class Generator(IGenerator):
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.client = OpenAI(base_url="http://host.docker.internal:8080", api_key="not-needed")

    def generate_answer(self, query: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        """
        Генерирует ответ на запрос с учётом контекста, полученного от ретривера.
        :param query: Запрос, для которого нужно сгенерировать ответ
        :param temperature: Температура для генерации (по умолчанию 0.2)
        :param max_tokens: Максимальное количество токенов для генерации (по умолчанию 600)
        :return: Сгенерированный ответ
        """

        # 1. Получаем контекст с использованием retriever
        context_list = self.retriever.find_similar_context(query)

        if not context_list:
            return "Нет релевантных контекстов."

        # 2. Получаем лучший контекст, используя лучший матч
        best_context = self.retriever.best_match([query], context_list, top_k=1)

        # 3. Отправляем запрос в OpenAI, чтобы получить сгенерированный ответ
        response = self.client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "Вы технический специалист службы поддержки корпоративных систем. Ваша задача - предоставить точный и понятный ответ на основе предоставленной документации." +
                                              "Требования к ответу: 1. Используйте ТОЛЬКО информацию из предоставленного контекста" +
                                            "2. Если вопрос требует пошаговых инструкций, структурируйте ответ в виде нумерованного списка" +
                                              "Если информации недостаточно или её нет в контексте, сообщите об этом: В предоставленной документации информация по данному вопросу отсутствует" +
                                              "4. Ответ должен быть конкретным и относиться только к заданному вопросу" +
                                              "5. Избегайте предположений и догадок"},
                {"role": "user", "content": f"Запрос пользователя: {query}\n\nКонтекст:\n{best_context}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()