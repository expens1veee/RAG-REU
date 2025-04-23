from fastapi import FastAPI
from src.internal.http_server.server import Server
from src.internal.storage.qdrant import QdrantStorage
from src.internal.retriever.retriever import Retriever
import uvicorn
from dataclasses import dataclass
import os
import logging
from src.storage_config.config import StorageConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    config = StorageConfig(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
        vector_size=384
    )
    logger.info(f"Storage config: host={config.host}, port={config.port}")

    storage = QdrantStorage(config)
    retriever = Retriever(storage=storage)

    # Создаём приложение FastAPI
    app = FastAPI()

    # Создаём экземпляр класса Server и подключаем его роутер
    server = Server(storage=storage, retriever=retriever)
    app.include_router(server.router)

    # Запускаем сервер
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
