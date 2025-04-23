from dataclasses import dataclass


@dataclass
class StorageConfig:
    host: str
    port: int
    vector_size: int = 384
    collection_name: str = "documents"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.host}:{self.port}"
