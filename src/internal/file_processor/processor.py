"""
PDF Reader для RAG с использованием PyMuPDF и LateChunker.
Этот скрипт извлекает текст из PDF файлов и разбивает его на чанки для дальнейшего использования в RAG системах.
"""

import os
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class PDFChunk:
    """Класс для представления чанка текста из PDF."""
    text: str
    page_number: int
    chunk_id: str
    metadata: Dict[str, Any]


class PDFReader:
    """Класс для извлечения текста из PDF файлов с помощью PyMuPDF."""

    def __init__(self, pdf_path: str):
        """
        Инициализация PDF Reader.

        Args:
            pdf_path: Путь к PDF файлу.
        """
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.filename = os.path.basename(pdf_path)

    def extract_text_from_page(self, page_num: int) -> str:
        """
        Извлекает текст из определенной страницы PDF.

        Args:
            page_num: Номер страницы для извлечения текста.

        Returns:
            Текст со страницы.
        """
        page = self.doc[page_num]
        return page.get_text()

    def extract_all_text(self) -> List[Tuple[int, str]]:
        """
        Извлекает текст из всего PDF документа.

        Returns:
            Список кортежей (номер_страницы, текст)
        """
        text_by_page = []
        for page_num in range(len(self.doc)):
            text = self.extract_text_from_page(page_num)
            text_by_page.append((page_num, text))
        return text_by_page

    def extract_metadata(self) -> Dict[str, Any]:
        """
        Извлекает метаданные из PDF документа.

        Returns:
            Словарь с метаданными документа.
        """
        metadata = self.doc.metadata
        metadata['page_count'] = len(self.doc)
        metadata['filename'] = self.filename
        return metadata

    def close(self):
        """Закрывает PDF документ."""
        self.doc.close()


class LateChunker:
    """
    Класс для разбиения текста на чанки с использованием LateChunker подхода.
    LateChunker создает чанки с учетом семантической целостности текста.
    """

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            separator: str = "\n",
            length_function=len
    ):
        """
        Инициализация LateChunker.

        Args:
            chunk_size: Целевой размер чанка.
            chunk_overlap: Размер перекрытия между соседними чанками.
            separator: Разделитель для разбиения текста на предложения или абзацы.
            length_function: Функция для определения длины текста.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.length_function = length_function

    def _split_text(self, text: str) -> List[str]:
        """
        Разбивает текст на сегменты с использованием разделителя.

        Args:
            text: Текст для разбиения.

        Returns:
            Список сегментов текста.
        """
        # Разбиваем текст по разделителю и удаляем пустые строки
        splits = text.split(self.separator)
        return [s for s in splits if s.strip()]

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Объединяет сегменты в чанки с учетом целевого размера и перекрытия.

        Args:
            splits: Список сегментов текста.

        Returns:
            Список чанков текста.
        """
        chunks = []
        current_chunk = []
        current_chunk_size = 0

        for split in splits:
            split_size = self.length_function(split)

            # Если текущий чанк пуст или добавление нового сегмента не превысит целевой размер
            if not current_chunk or current_chunk_size + split_size <= self.chunk_size:
                current_chunk.append(split)
                current_chunk_size += split_size
            else:
                # Сохраняем текущий чанк и начинаем новый
                chunks.append(self.separator.join(current_chunk))

                # Определяем, сколько последних сегментов нужно перенести в новый чанк для перекрытия
                overlap_size = 0
                overlap_chunks = []

                for i in range(len(current_chunk) - 1, -1, -1):
                    if overlap_size < self.chunk_overlap:
                        overlap_size += self.length_function(current_chunk[i])
                        overlap_chunks.insert(0, current_chunk[i])
                    else:
                        break

                # Начинаем новый чанк с перекрытием
                current_chunk = overlap_chunks + [split]
                current_chunk_size = overlap_size + split_size

        # Добавляем последний чанк, если он не пустой
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))

        return chunks

    def create_chunks(self, text: str) -> List[str]:
        """
        Создает чанки из текста с учетом семантической целостности.

        Args:
            text: Исходный текст.

        Returns:
            Список чанков текста.
        """
        splits = self._split_text(text)
        chunks = self._merge_splits(splits)
        return chunks


class PDFChunker:
    """
    Класс для разбиения PDF документов на чанки для использования в RAG.
    Объединяет функциональность PDFReader и LateChunker.
    """

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            separator: str = "\n"
    ):
        """
        Инициализация PDFChunker.

        Args:
            chunk_size: Целевой размер чанка.
            chunk_overlap: Размер перекрытия между соседними чанками.
            separator: Разделитель для разбиения текста.
        """
        self.chunker = LateChunker(chunk_size, chunk_overlap, separator)

    def process_pdf(self, pdf_path: str) -> List[PDFChunk]:
        """
        Обрабатывает PDF файл и создает чанки.

        Args:
            pdf_path: Путь к PDF файлу.

        Returns:
            Список чанков с метаданными.
        """
        reader = PDFReader(pdf_path)
        doc_metadata = reader.extract_metadata()

        all_chunks = []
        text_by_page = reader.extract_all_text()

        for page_num, page_text in text_by_page:
            # Создаем чанки для текущей страницы
            chunks = self.chunker.create_chunks(page_text)

            # Создаем объекты PDFChunk для каждого чанка
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{os.path.basename(pdf_path)}_p{page_num + 1}_c{i + 1}"

                # Собираем метаданные для чанка
                metadata = {
                    "source": pdf_path,
                    "page": page_num + 1,
                    "chunk_index": i,
                    "doc_metadata": doc_metadata
                }

                chunk = chunk_text

                all_chunks.append(chunk)

        reader.close()
        return all_chunks

    def save_chunks_to_qdrant(self, chunks: List[PDFChunk], output_path: str):
        """
        Сохраняет чанки в qdrant.

        Args:
            chunks: Список чанков.
            output_path: Путь для сохранения JSON файла.
        """
        # Преобразуем объекты PDFChunk в словари
        chunks_dict = [asdict(chunk) for chunk in chunks]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_dict, f, ensure_ascii=False, indent=2)


def process_pdf_directory(directory_path: str, output_directory: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Обрабатывает все PDF файлы в указанной директории.

    Args:
        directory_path: Путь к директории с PDF файлами.
        output_directory: Путь к директории для сохранения результатов.
        chunk_size: Размер чанка.
        chunk_overlap: Размер перекрытия между чанками.
    """
    # Создаем директорию для выходных данных, если она не существует
    os.makedirs(output_directory, exist_ok=True)

    # Инициализируем чанкер
    chunker = PDFChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Находим все PDF файлы в директории
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]

    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        pdf_path = os.path.join(directory_path, pdf_file)
        output_file = os.path.join(output_directory, f"{os.path.splitext(pdf_file)[0]}_chunks.json")

        try:
            # Обрабатываем PDF и сохраняем чанки
            chunks = chunker.process_pdf(pdf_path)
            chunker.save_chunks_to_qdrant(chunks, output_file)
            print(f"Processed {pdf_file}: Created {len(chunks)} chunks")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")


# Пример обработки директории с PDF файлами
# process_pdf_directory("pdf_directory", "output_directory", chunk_size=800, chunk_overlap=150)