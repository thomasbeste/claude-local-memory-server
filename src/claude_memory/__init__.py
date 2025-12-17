"""Claude Memory - Local or central persistent memory for Claude via MCP."""

from .storage import MemoryStorage
from .server import create_server, run

__version__ = "0.1.0"
__all__ = ["MemoryStorage", "create_server", "run"]

# Optional imports for server mode
try:
    from .api import app, create_app, run_server
    __all__.extend(["app", "create_app", "run_server"])
except ImportError:
    pass  # Server dependencies not installed
