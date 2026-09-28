from typing import Generator, Dict, Any
from warcio.archiveiterator import ArchiveIterator as WarcioIterator
from fastwarc.warc import ArchiveIterator as FastWarcIterator, WarcRecordType # type: ignore
import io

def parse_warc_fastwarc(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Yields raw byte payload and content-type from WARC records using FastWARC."""
    with open(file_path, 'rb') as stream:
        for record in FastWarcIterator(stream, record_types=WarcRecordType.response):
            if record.http_headers:
                content_type = record.http_headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    yield {
                        "url": record.headers.get('WARC-Target-URI'),
                        "warc_date": record.headers.get('WARC-Date'),
                        "content_type": content_type,
                        "record_id": record.headers.get('WARC-Record-ID'),
                        "raw_bytes": record.reader.read()
                    }

def parse_warc_warcio(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Yields raw byte payload and content-type from WARC records using warcio."""
    with open(file_path, 'rb') as stream:
        for record in WarcioIterator(stream):
            if record.rec_type == 'response':
                content_type = record.http_headers.get_header('Content-Type', '') if record.http_headers else ''
                if 'text/html' in content_type:
                    yield {
                        "url": record.rec_headers.get_header('WARC-Target-URI'),
                        "warc_date": record.rec_headers.get_header('WARC-Date'),
                        "content_type": content_type,
                        # FIX: warcio record objects don't have a `.headers`
                        # attribute (only `.rec_headers` and `.http_headers`).
                        # The old line `record.headers.get('WARC-Record-ID')`
                        # would raise AttributeError on every single record,
                        # crashing this generator the moment it's used.
                        "record_id": record.rec_headers.get_header('WARC-Record-ID'),
                        "raw_bytes": record.content_stream().read()
                    }

def parse_warc_stream_fastwarc(stream) -> Generator[Dict[str, Any], None, None]:
    """Yields raw byte payload from live network stream using FastWARC with in-memory buffer."""
    # StreamBytesIO Buffer seek
    stream_buffer = io.BytesIO(stream.read())

    for record in FastWarcIterator(stream_buffer, record_types=WarcRecordType.response):
        if record.http_headers:
            content_type = record.http_headers.get('Content-Type', '')
            if 'text/html' in content_type:
                yield {
                    "url": record.headers.get('WARC-Target-URI'),
                    "warc_date": record.headers.get('WARC-Date'),
                    "content_type": content_type,
                    "record_id": record.headers.get('WARC-Record-ID'),
                    "raw_bytes": record.reader.read()
                }