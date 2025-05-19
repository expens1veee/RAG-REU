from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Literal
from src.internal.storage.qdrant import QdrantStorage
from src.internal.retriever.retriever import Retriever
from src.internal.generator.generator import Generator
from src.internal.file_processor.processor import PDFChunker


class AskRequest(BaseModel):
    query: str
    token: Optional[str] = None  # Опциональное поле для токена


class FeedbackRequest(BaseModel):
    message_id: str
    feedback_type: Literal['like', 'dislike']


class Server:
    def __init__(self, storage: QdrantStorage, retriever: Retriever = None, generator: Generator = None):
        self.storage = storage
        self.retriever = retriever
        self.generator = generator
        # Создаём роутер с префиксом /api
        self.router = APIRouter(
            prefix="/api",
            tags=["api"],
        )
        # Регистрируем маршруты
        self._register_routes()

    def _register_routes(self):
        # Эндпоинт для проверки состояния сервера
        @self.router.get("/status")
        async def status():
            return {"message": "Alive"}

        # Эндпоинт для обработки запросов от пользователя
        @self.router.post("/ask")
        async def ask(request: AskRequest):
            query = request.query
            return self.generator.generate_answer(query)

        @self.router.post("/feedback")
        async def feedback(request: FeedbackRequest):
            """
            Эндпоинт для обработки обратной связи пользователя
            """
            try:
                # Здесь можно добавить логику сохранения фидбека
                # Например, в базу данных или файл
                print(f"Получен фидбек: {request.feedback_type} для сообщения {request.message_id}")
                
                # Если используете генератор, можно передать фидбек ему
                if self.generator:
                    self.generator.process_feedback(request.message_id, request.feedback_type)
                
                return {"status": "success", "message": "Спасибо за обратную связь!"}
            except Exception as e:
                print(f"Ошибка при обработке фидбека: {e}")
                return {"status": "error", "message": "Не удалось обработать обратную связь"}

        @self.router.post("/debug/create-collection")
        async def create_collection():
            from qdrant_client.http.models import Distance, VectorParams

            if self.storage.client.collection_exists(self.storage.collection_name):
                return {"message": f"Коллекция '{self.storage.collection_name}' уже существует"}

            self.storage.client.create_collection(
                collection_name=self.storage.collection_name,
                vectors_config=VectorParams(
                    size=self.storage.vector_size,
                    distance=Distance.COSINE
                )
            )
            return {"message": f"Коллекция '{self.storage.collection_name}' успешно создана"}

        @self.router.get("/load-documents")
        async def load_documents():
            """
            Эндпоинт для загрузки и обработки документов в Qdrant.
            """
            pdf_paths = ["/app/src/internal/file_processor/приложение о курсовых.pdf", "/app/src/internal/file_processor/проход.pdf", "/app/src/internal/file_processor/экзамены.pdf"]
            for pdf_path in pdf_paths:
                chunker = PDFChunker(chunk_size=800, chunk_overlap=150)

                chunks = chunker.process_pdf(pdf_path)
                print(f"Created {len(chunks)} chunks from {pdf_path}")
                metadata = [{"source": "тест"}] * len(chunks)
                self.retriever.generate_embeddings(chunks, metadata)





