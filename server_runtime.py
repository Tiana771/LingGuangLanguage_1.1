"""Prevent two local LingGuang launchers from sharing a development port."""
from contextlib import contextmanager
from pathlib import Path
import os
import socket


class ServerAlreadyRunningError(RuntimeError):
    pass


@contextmanager
def claim_server_port(host, port, *, lock_dir=None):
    root = Path(lock_dir) if lock_dir is not None else Path(__file__).resolve().parent
    handle = (root / f".linguang-server-{port}.lock").open("a+b")
    acquired = False
    try:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            raise ServerAlreadyRunningError(f"{port}端口的系统已经在运行。请先在原终端按 Ctrl+C 停止，再启动。") from None
        # Windows Werkzeug permits address reuse. Test exclusive ownership first,
        # including legacy servers that were started without our lock file.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                probe.bind((host, port))
            except OSError:
                raise ServerAlreadyRunningError(f"{port}端口已被占用。未启动第二个服务，也未停止其他进程。") from None
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
