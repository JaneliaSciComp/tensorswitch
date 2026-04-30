"""Unit tests for retry logic in chunk write operations."""

import pytest
from unittest.mock import MagicMock, patch

from tensorswitch_v2.core.retry import retry_write, MAX_RETRIES, BACKOFF_BASE


class TestRetryWrite:
    """Tests for the retry_write helper."""

    def test_success_no_retry(self):
        """Write succeeds on first attempt — no retry needed."""
        fn = MagicMock()
        retry_write(fn, chunk_id=0)
        fn.assert_called_once()

    @patch("tensorswitch_v2.core.retry.time.sleep")
    def test_success_after_transient_oserror(self, mock_sleep):
        """OSError on first attempt, success on second."""
        fn = MagicMock(side_effect=[OSError("NFS timeout"), None])
        retry_write(fn, chunk_id=5, verbose=False)
        assert fn.call_count == 2
        mock_sleep.assert_called_once_with(BACKOFF_BASE * (4 ** 0))  # 1s

    @patch("tensorswitch_v2.core.retry.time.sleep")
    def test_success_after_two_failures(self, mock_sleep):
        """Two transient failures, then success on third attempt."""
        fn = MagicMock(side_effect=[
            IOError("disk busy"),
            TimeoutError("write timed out"),
            None,
        ])
        retry_write(fn, chunk_id=10, verbose=False)
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("tensorswitch_v2.core.retry.time.sleep")
    def test_permanent_failure_raises(self, mock_sleep):
        """All retries exhausted — raises the original OSError."""
        fn = MagicMock(side_effect=OSError("persistent NFS error"))
        with pytest.raises(OSError, match="persistent NFS error"):
            retry_write(fn, chunk_id=0, verbose=False)
        assert fn.call_count == MAX_RETRIES + 1
        assert mock_sleep.call_count == MAX_RETRIES

    def test_non_io_error_no_retry(self):
        """ValueError is raised immediately without retry."""
        fn = MagicMock(side_effect=ValueError("shape mismatch"))
        with pytest.raises(ValueError, match="shape mismatch"):
            retry_write(fn, chunk_id=0, verbose=False)
        fn.assert_called_once()

    def test_keyboard_interrupt_propagates(self):
        """KeyboardInterrupt is never caught or retried."""
        fn = MagicMock(side_effect=KeyboardInterrupt)
        with pytest.raises(KeyboardInterrupt):
            retry_write(fn, chunk_id=0, verbose=False)
        fn.assert_called_once()

    @patch("tensorswitch_v2.core.retry.time.sleep")
    def test_backoff_schedule(self, mock_sleep):
        """Verify exponential backoff: 1s, 4s, 16s."""
        fn = MagicMock(side_effect=OSError("fail"))
        with pytest.raises(OSError):
            retry_write(fn, chunk_id=0, verbose=False)
        expected_waits = [BACKOFF_BASE * (4 ** i) for i in range(MAX_RETRIES)]
        actual_waits = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_waits == expected_waits

    @patch("tensorswitch_v2.core.retry.time.sleep")
    def test_verbose_prints_retry_message(self, mock_sleep, capsys):
        """Verbose mode prints retry information."""
        fn = MagicMock(side_effect=[OSError("NFS timeout"), None])
        retry_write(fn, chunk_id=42, verbose=True)
        captured = capsys.readouterr()
        assert "Retry 1/3" in captured.out
        assert "chunk 42" in captured.out
        assert "OSError" in captured.out

    def test_timeout_error_is_retried(self):
        """TimeoutError triggers retry (important for network storage)."""
        fn = MagicMock(side_effect=[TimeoutError("connection timeout"), None])
        with patch("tensorswitch_v2.core.retry.time.sleep"):
            retry_write(fn, chunk_id=0, verbose=False)
        assert fn.call_count == 2
