"""Opt-in per-socket buffers; never modifies sysctl or management interfaces.

Linux-only FORCE options require CAP_NET_ADMIN. Buffers must be set BEFORE
connect/listen so TCP window scaling is negotiated with the larger window.
httpcore's ordinary socket_options run after connect in the pinned version.
"""
import socket
import httpx
from httpcore._backends.sync import SyncBackend, SyncStream
from httpcore import ConnectError, ConnectTimeout


def set_buffers(sock, mib):
    if type(mib) is not int or not 1<=mib<=64:
        raise ValueError("TCP buffer must be 1..64 MiB")
    for option,normal in ((32,socket.SO_SNDBUF),(33,socket.SO_RCVBUF)):
        sock.setsockopt(socket.SOL_SOCKET,option,mib*1024**2)
        if sock.getsockopt(socket.SOL_SOCKET,normal)<mib*1024**2:
            raise RuntimeError("Kernel did not accept requested socket buffer")


class PreconnectBackend(SyncBackend):
    def __init__(self,mib):
        self.mib=mib

    def connect_tcp(self,host,port,timeout=None,local_address=None,socket_options=None):
        sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout)
            for option in socket_options or ():
                sock.setsockopt(*option)
            set_buffers(sock,self.mib)
            sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            if local_address:
                sock.bind((local_address,0))
            sock.connect((host,port))
            return SyncStream(sock)
        except socket.timeout as error:
            sock.close()
            raise ConnectTimeout(str(error)) from error
        except OSError as error:
            sock.close()
            raise ConnectError(str(error)) from error
        except BaseException:
            sock.close()
            raise


def http_client(mib=0,**kwargs):
    if mib:
        transport=httpx.HTTPTransport(trust_env=False)
        if not isinstance(transport._pool._network_backend,SyncBackend):
            raise RuntimeError("Unsupported httpcore backend; revalidate pinned transport")
        transport._pool._network_backend=PreconnectBackend(mib)
        kwargs["transport"]=transport
    return httpx.Client(**kwargs)


def serve(app,host,port,mib=0):
    import uvicorn
    if not mib:
        return uvicorn.run(app,host=host,port=port,access_log=False)
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        set_buffers(sock,mib)
        sock.bind((host,port))
        sock.listen(128)
        config=uvicorn.Config(app,host=host,port=port,access_log=False)
        uvicorn.Server(config).run(sockets=[sock])
