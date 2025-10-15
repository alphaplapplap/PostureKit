"""
PostgreSQL lifecycle manager for PostureKit
Starts PostgreSQL when app starts, stops when app exits
"""

import subprocess
import time
import socket
import atexit
from pathlib import Path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PostgresManager:
    """Manages PostgreSQL server lifecycle"""

    def __init__(self, data_dir: str = "/opt/homebrew/var/postgresql@16", port: int = 5432):
        self.data_dir = data_dir
        self.port = port
        self.pg_ctl = "/opt/homebrew/opt/postgresql@16/bin/pg_ctl"
        self._process = None

    def is_running(self) -> bool:
        """Check if PostgreSQL is already running on the configured port"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', self.port))
                return result == 0
        except Exception:
            return False

    def start(self, wait: bool = True, timeout: int = 30) -> bool:
        """
        Start PostgreSQL server if not already running

        Args:
            wait: Wait for server to be ready
            timeout: Maximum wait time in seconds

        Returns:
            True if started successfully or already running
        """
        if self.is_running():
            logger.info(f"PostgreSQL already running on port {self.port}")
            return True

        logger.info(f"Starting PostgreSQL from {self.data_dir}...")

        try:
            cmd = [
                self.pg_ctl,
                "-D", self.data_dir,
                "-l", "/opt/homebrew/var/log/postgresql@16.log",
                "start"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                logger.error(f"Failed to start PostgreSQL: {result.stderr}")
                return False

            if wait:
                return self._wait_for_ready(timeout)

            logger.info("PostgreSQL started successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"PostgreSQL start timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Failed to start PostgreSQL: {e}")
            return False

    def _wait_for_ready(self, timeout: int = 30) -> bool:
        """Wait for PostgreSQL to be ready to accept connections"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_running():
                logger.info("PostgreSQL is ready")
                return True
            time.sleep(0.5)

        logger.error(f"PostgreSQL not ready after {timeout}s")
        return False

    def stop(self, mode: str = "fast") -> bool:
        """
        Stop PostgreSQL server

        Args:
            mode: Shutdown mode - 'smart', 'fast', or 'immediate'

        Returns:
            True if stopped successfully
        """
        if not self.is_running():
            logger.info("PostgreSQL not running")
            return True

        logger.info(f"Stopping PostgreSQL (mode: {mode})...")

        try:
            cmd = [
                self.pg_ctl,
                "-D", self.data_dir,
                "-m", mode,
                "stop"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.warning(f"PostgreSQL stop returned non-zero: {result.stderr}")

            logger.info("PostgreSQL stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop PostgreSQL: {e}")
            return False

    def restart(self) -> bool:
        """Restart PostgreSQL server"""
        logger.info("Restarting PostgreSQL...")
        self.stop()
        time.sleep(2)
        return self.start()


# Global instance
_pg_manager = None


def get_postgres_manager() -> PostgresManager:
    """Get or create global PostgresManager instance"""
    global _pg_manager
    if _pg_manager is None:
        _pg_manager = PostgresManager()
        # Register cleanup on exit
        atexit.register(_pg_manager.stop)
    return _pg_manager


def ensure_postgres_running() -> bool:
    """Ensure PostgreSQL is running, start if needed"""
    manager = get_postgres_manager()
    return manager.start()


def stop_postgres():
    """Stop PostgreSQL if managed by this application"""
    manager = get_postgres_manager()
    manager.stop()
