from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.internal.http_server.server import Server
from src.internal.storage.qdrant import QdrantStorage
from src.internal.retriever.retriever import Retriever
import uvicorn
from dataclasses import dataclass
import os
import logging
from src.storage_config.config import StorageConfig
from src.internal.generator.generator import Generator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    config = StorageConfig(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
        vector_size=312
    )
    logger.info(f"Storage config: host={config.host}, port={config.port}")

    storage = QdrantStorage(config)
    retriever = Retriever(storage=storage)

    # Создаём приложение FastAPI
    app = FastAPI()

    # Монтируем статические файлы
    app.mount("/static", StaticFiles(directory="src/internal/http_server/static"), name="static")

    # Создание генератора с переданным retriever
    generator = Generator(retriever=retriever)

    # Создаём экземпляр класса Server и подключаем его роутер
    server = Server(storage=storage, retriever=retriever, generator=generator)
    app.include_router(server.router)

    @app.get("/")
    async def read_root():
        return FileResponse("src/internal/http_server/static/index.html")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # Запускаем сервер
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
