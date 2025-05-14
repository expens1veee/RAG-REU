from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from src.internal.storage.qdrant import QdrantStorage
from src.internal.retriever.retriever import Retriever
from src.internal.generator.generator import Generator
from src.internal.file_processor.processor import PDFChunker


class AskRequest(BaseModel):
    query: str
    token: Optional[str] = None  # Опциональное поле для токена


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

        @self.router.post("/debug/save-test")
        async def save_test():
            chunks = ["Кошка сидит на дереве", "Собака лает на прохожего", "Погода сегодня хорошая"]
            metadata = [{"source": "тест"}] * len(chunks)
            self.retriever.generate_embeddings(chunks, metadata)
            return {"message": "Сохранено"}

        @self.router.post("/debug/find-similar")
        async def find_similar():
            query = "Что делает кошка?"
            context_list = self.retriever.find_similar_context(query)

            if not context_list:
                return {"result": None, "message": "Нет релевантных контекстов"}

            best = self.retriever.best_match([query], context_list, top_k=1)
            return {"result": best}

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
            pdf_path = "/app/src/internal/file_processor/приложение о курсовых.pdf"
            chunker = PDFChunker(chunk_size=800, chunk_overlap=150)

            chunks = chunker.process_pdf(pdf_path)
            print(f"Created {len(chunks)} chunks from {pdf_path}")
            metadata = [{"source": "тест"}] * len(chunks)
            self.retriever.generate_embeddings(chunks, metadata)





