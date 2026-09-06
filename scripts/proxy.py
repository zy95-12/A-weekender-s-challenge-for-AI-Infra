"""Loopback-only byte proxy into the Enterprise namespace (preserves SSE)."""
import select
import socket
import socketserver


class Proxy(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            with socket.create_connection(("10.204.0.2", 8000), timeout=10) as remote:
                remote.settimeout(None)
                sockets = [self.request, remote]
                while True:
                    readable, _, _ = select.select(sockets, [], [], 300)
                    if not readable:
                        return
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        (remote if source is self.request else self.request).sendall(data)
        except OSError:
            return


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server(("127.0.0.1", 8000), Proxy) as server:
        server.serve_forever()
