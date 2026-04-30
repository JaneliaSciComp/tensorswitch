"""Retry helper for chunk write operations.

Provides exponential-backoff retry for transient I/O errors (NFS/Lustre
timeouts, network storage glitches) that can occur during large parallel
chunk writes. Non-I/O errors (ValueError, shape mismatches) are raised
immediately without retry.
"""

import time

MAX_RETRIES = 3
BACKOFF_BASE = 1  # seconds: attempts sleep 1s, 4s, 16s


def retry_write(write_fn, chunk_id, verbose=True):
    """Call write_fn() with up to MAX_RETRIES retries on I/O errors.

    Uses exponential backoff: 1s, 4s, 16s between attempts.

    Args:
        write_fn: Zero-argument callable that performs the chunk write.
        chunk_id: Chunk index for log messages.
        verbose: Whether to print retry messages.

    Raises:
        The original exception if all retries are exhausted, or immediately
        for non-I/O errors.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            write_fn()
            return
        except KeyboardInterrupt:
            raise
        except (OSError, IOError, TimeoutError) as e:
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE * (4 ** attempt)
                if verbose:
                    print(
                        f"  Retry {attempt + 1}/{MAX_RETRIES} for chunk {chunk_id} "
                        f"after {type(e).__name__}: {e} (waiting {wait}s)"
                    )
                time.sleep(wait)
            else:
                raise
        except Exception:
            # Non-I/O errors (shape mismatch, dtype, etc.) — don't retry
            raise
