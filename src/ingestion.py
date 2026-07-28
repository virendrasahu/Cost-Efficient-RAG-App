import os
import hashlib
from typing import List, Dict, Any, Set
from pypdf import PdfReader
from bs4 import BeautifulSoup
import markdown

from src.logger import logger
from src.config import settings

def generate_chunk_id(source_path: str, chunk_index: int, text: str) -> str:
    """Generate a deterministic unique ID (SHA256) for a text chunk."""
    raw_identifier = f"{source_path}_{chunk_index}_{text}"
    return hashlib.sha256(raw_identifier.encode('utf-8')).hexdigest()

class DocumentLoader:
    """Loads documents of various formats (PDF, HTML, MD, TXT)."""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text

    @staticmethod
    def load_html(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            # Extract plain text from body or whole document
            return soup.get_text(separator=' ', strip=True)

    @staticmethod
    def load_markdown(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            # Convert to html then clean text, or simply return plain text
            html_text = markdown.markdown(raw_text)
            soup = BeautifulSoup(html_text, 'html.parser')
            return soup.get_text(separator=' ', strip=True)

    @staticmethod
    def load_document(file_path: str) -> tuple[str, str]:
        """Load document based on file extension. Returns (text_content, file_type)."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return DocumentLoader.load_pdf(file_path), 'pdf'
        elif ext in ['.html', '.htm']:
            return DocumentLoader.load_html(file_path), 'html'
        elif ext in ['.md', '.markdown', '.txt']:
            return DocumentLoader.load_markdown(file_path), 'md'
        else:
            raise ValueError(f"Unsupported file format: {ext}")

class RecursiveCharacterTextSplitter:
    """Splits text into chunks using recursive separator matching."""
    def __init__(self, chunk_size: int = settings.DEFAULT_CHUNK_SIZE, chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            if end < text_len:
                # Try to split at a separator
                for sep in self.separators:
                    if sep == "":
                        break
                    idx = text.rfind(sep, start, end)
                    if idx != -1 and idx > start:
                        end = idx + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start pointer forward with overlap
            if end >= text_len:
                break
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

def process_and_deduplicate(
    file_path: str,
    existing_ids: Set[str],
    chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """Loads a document, chunks it, generates SHA256 IDs, and deduplicates against existing vector store IDs."""
    logger.info(f"Processing document: {file_path}")
    text_content, file_type = DocumentLoader.load_document(file_path)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_text(text_content)
    
    new_records = []
    skipped_count = 0
    
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_id = generate_chunk_id(file_path, idx, chunk_text)
        if chunk_id in existing_ids:
            skipped_count += 1
            continue
        
        new_records.append({
            "id": chunk_id,
            "text": chunk_text,
            "source": file_path,
            "chunk_index": idx,
            "file_type": file_type
        })
        
    logger.info(f"Document {file_path}: Created {len(new_records)} new chunks, skipped {skipped_count} existing chunks.")
    return new_records
