"""
HTTP helpers for the bulk catalog sources (download step only).

Two pieces, both built on httpx:

- ``PoliteHttpClient``: a small wrapper that every API call of the catalog
  builder goes through. It sends a generic User-Agent (no personal data),
  spaces requests with a shared rate limit, honours ``Retry-After`` (HTTP 429,
  503 and the MediaWiki ``maxlag`` error) and retries transient failures with
  exponential backoff.
- ``download_file``: a resumable file download. Large dumps can be split into
  byte ranges fetched in parallel (archive.org throttles each connection);
  every range is written to its own ``.partN`` file, so an interrupted run
  continues where it stopped. The parts are joined once all are complete.

Parsing never touches the network: the source adapters read the files these
helpers leave in ``data/downloads/``.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Generic identification: no person name, e-mail address or repository name.
USER_AGENT = "book-catalog-builder/0.1 (academic project; python-httpx)"

# httpx advertises "zstd" when the zstandard package is installed; Wikimedia's
# edge answers such requests with HTTP 403 (robot policy), so ask for the
# encodings every server supports.
BASE_HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

RETRY_STATUS = {429, 500, 502, 503, 504}

STALL_WINDOW_S = 60.0
"""Seconds over which a range download's speed is checked"""

MIN_BYTES_PER_S = 1024
"""Below this average speed over a window, a range connection is reopened"""


class RateLimiter:
    """Thread-safe limiter: at most ``rate`` request starts per second overall."""

    def __init__(self, rate_per_second: float) -> None:
        self._interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> None:
        """Block until the caller may start a request."""
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class PoliteHttpClient:
    """
    GET/POST with a shared rate limit, ``Retry-After`` and exponential backoff.

    Args:
        rate_per_second: Maximum request starts per second across all threads
        max_retries: Attempts after the first one for transient failures
        timeout_s: Read timeout for one request
        client: Optional httpx.Client (tests inject a mock transport)
    """

    def __init__(
        self,
        rate_per_second: float = 2.0,
        max_retries: int = 6,
        timeout_s: float = 90.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._limiter = RateLimiter(rate_per_second)
        self._max_retries = max_retries
        self._client = client or httpx.Client(
            headers=BASE_HEADERS,
            timeout=httpx.Timeout(timeout_s, connect=20.0),
            follow_redirects=True,
        )
        self.retry_events = 0
        """Number of retried requests (rate-limit and transient errors)"""

        self.rate_limit_events = 0
        """Number of HTTP 429 / maxlag answers received"""

    def request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """
        Send one request, retrying transient failures.

        Returns:
            The successful response (status < 400)

        Raises:
            RuntimeError: If every attempt failed
        """
        last_error = ""
        for attempt in range(self._max_retries + 1):
            self._limiter.wait()
            try:
                response = self._client.request(
                    method, url, params=params, data=data, headers=headers
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._sleep_backoff(attempt, None)
                continue

            if response.status_code < 400 and not _is_maxlag(response):
                return response

            if response.status_code == 429 or _is_maxlag(response):
                self.rate_limit_events += 1
            if response.status_code in RETRY_STATUS or _is_maxlag(response):
                last_error = f"HTTP {response.status_code}"
                self._sleep_backoff(attempt, response.headers.get("Retry-After"))
                continue

            raise RuntimeError(f"HTTP {response.status_code} for {url}: {response.text[:200]}")

        raise RuntimeError(f"Giving up on {url} after {self._max_retries + 1} attempts ({last_error})")

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> httpx.Response:
        """GET with retries."""
        return self.request("GET", url, params=params, **kwargs)

    def post(self, url: str, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> httpx.Response:
        """POST (form-encoded) with retries."""
        return self.request("POST", url, data=data, **kwargs)

    def _sleep_backoff(self, attempt: int, retry_after: Optional[str]) -> None:
        """Sleep before a retry: Retry-After if the server sent one, else 2^attempt s."""
        self.retry_events += 1
        delay = min(2.0 ** attempt, 120.0)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        logger.warning("Retrying in %.1f s (attempt %d)", delay, attempt + 1)
        time.sleep(delay)

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()


def _is_maxlag(response: httpx.Response) -> bool:
    """True if a MediaWiki API answered with the 'maxlag' error (server lagged)."""
    if response.headers.get("X-Database-Lag") is None and "maxlag" not in response.text[:300]:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("error", {}).get("code") == "maxlag"


def download_file(
    url: str,
    dest: Path,
    segments: int = 1,
    timeout_s: float = 120.0,
    max_retries: int = 50,
) -> Path:
    """
    Download ``url`` to ``dest``, resuming any previous partial download.

    With ``segments > 1`` and a server that accepts byte ranges, the file is
    split into that many ranges downloaded in parallel into ``dest.partN``
    files, then joined. A file that already exists at ``dest`` with the
    expected size is not downloaded again.

    Args:
        url: Source URL (redirects are followed)
        dest: Final path of the file
        segments: Number of parallel byte ranges (1 = a single stream)
        timeout_s: Read timeout per request
        max_retries: Retries per range before giving up

    Returns:
        The path of the complete file
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=BASE_HEADERS, follow_redirects=True, timeout=timeout_s) as client:
        head = client.head(url)
        head.raise_for_status()
        total = int(head.headers.get("Content-Length", "0") or 0)
        final_url = str(head.url)
        ranges_ok = head.headers.get("Accept-Ranges", "").lower() == "bytes"

    if dest.exists() and total and dest.stat().st_size == total:
        logger.info("Already downloaded: %s (%d bytes)", dest, total)
        return dest

    if not total or not ranges_ok:
        segments = 1
    # A resumed download must keep the byte ranges it started with, whatever
    # ``segments`` is now: the number of part files on disk decides.
    existing_parts = list(dest.parent.glob(f"{dest.name}.part*"))
    if existing_parts and total and ranges_ok:
        if segments != len(existing_parts):
            logger.info("Resuming %s with its original %d segment(s)", dest.name, len(existing_parts))
        segments = len(existing_parts)
    bounds = _segment_bounds(total, segments) if total else [(0, None)]
    parts = [dest.with_name(f"{dest.name}.part{i}") for i in range(len(bounds))]

    logger.info("Downloading %s -> %s (%d bytes, %d segment(s))", final_url, dest, total, len(bounds))
    with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
        futures = [
            pool.submit(_download_range, final_url, part, start, end, timeout_s, max_retries)
            for part, (start, end) in zip(parts, bounds)
        ]
        for future in futures:
            future.result()

    tmp = dest.with_name(dest.name + ".joining")
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as src:
                while chunk := src.read(1 << 22):
                    out.write(chunk)
    if total and tmp.stat().st_size != total:
        raise RuntimeError(f"Size mismatch for {dest}: {tmp.stat().st_size} != {total}")
    tmp.replace(dest)
    for part in parts:
        part.unlink()
    logger.info("Download complete: %s", dest)
    return dest


def _segment_bounds(total: int, segments: int) -> list[tuple[int, Optional[int]]]:
    """Split [0, total) into ``segments`` inclusive byte ranges."""
    size = -(-total // segments)
    return [(start, min(start + size, total) - 1) for start in range(0, total, size)]


def _download_range(
    url: str, part: Path, start: int, end: Optional[int], timeout_s: float, max_retries: int
) -> None:
    """Fetch bytes [start, end] into ``part``, appending to what is already there."""
    expected = None if end is None else end - start + 1
    failures = 0  # consecutive attempts that received nothing
    last_size = -1
    for _ in range(max_retries * 20):
        have = part.stat().st_size if part.exists() else 0
        if expected is not None and have >= expected:
            return
        if have > last_size:
            failures = 0  # the previous attempt made progress: reset the backoff
        last_size = have
        if failures > max_retries:
            break
        range_header = f"bytes={start + have}-" + ("" if end is None else str(end))
        try:
            with httpx.Client(
                headers={**BASE_HEADERS, "Range": range_header},
                follow_redirects=True,
                timeout=timeout_s,
            ) as client:
                with client.stream("GET", url) as response:
                    if response.status_code not in (200, 206):
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}", request=response.request, response=response
                        )
                    if response.status_code == 200 and have:
                        # Server ignored the range: restart this part from scratch.
                        part.unlink()
                    with open(part, "ab") as out:
                        window_start, window_bytes = time.monotonic(), 0
                        for chunk in response.iter_bytes():
                            out.write(chunk)
                            window_bytes += len(chunk)
                            elapsed = time.monotonic() - window_start
                            if elapsed >= STALL_WINDOW_S:
                                if window_bytes < MIN_BYTES_PER_S * elapsed:
                                    # A connection that trickles bytes never hits the
                                    # read timeout: drop it and reconnect.
                                    raise httpx.ReadTimeout("transfer stalled", request=response.request)
                                window_start, window_bytes = time.monotonic(), 0
            if expected is None:
                return
        except httpx.HTTPError as exc:
            failures += 1
            delay = min(2.0 ** min(failures, 6), 60.0)
            logger.warning("Range %s of %s failed (%s); retry in %.0f s", range_header, url, exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"Could not download range {start}-{end} of {url}")
