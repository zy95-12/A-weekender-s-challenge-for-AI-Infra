"""Bounded JSON header + raw tensor bytes; never pickle on the network."""
import json
import math
import struct
import numpy as np

MAX_BODY = 160 * 1024 * 1024
MAX_HEADER = 1024 * 1024


def pack(meta, tensors=()):
    arrays = [np.ascontiguousarray(t) for t in tensors]
    descriptors = [{"shape": list(a.shape), "dtype": a.dtype.str,
                    "bytes": a.nbytes} for a in arrays]
    header = json.dumps({"meta": meta, "tensors": descriptors},
                        separators=(",", ":")).encode()
    if len(header) > MAX_HEADER:
        raise ValueError("Header too large")
    body = struct.pack("!I", len(header)) + header + b"".join(a.tobytes() for a in arrays)
    if len(body) > MAX_BODY:
        raise ValueError("Body too large")
    return body


def unpack(body):
    if not 4 <= len(body) <= MAX_BODY:
        raise ValueError("Invalid body length")
    size, = struct.unpack("!I", body[:4])
    if not 0 < size <= MAX_HEADER or 4 + size > len(body):
        raise ValueError("Invalid header")
    header = json.loads(body[4:4 + size])
    if set(header) != {"meta", "tensors"} or len(header["tensors"]) > 2:
        raise ValueError("Invalid envelope")
    offset, arrays = 4 + size, []
    for desc in header["tensors"]:
        if desc["dtype"] != "<f2" or len(desc["shape"]) != 2:
            raise ValueError("Expected rank-2 FP16 tensor")
        if any(type(x) is not int or x < 1 for x in desc["shape"]):
            raise ValueError("Invalid shape")
        count = math.prod(desc["shape"]) * 2
        if count != desc["bytes"] or offset + count > len(body):
            raise ValueError("Invalid tensor length")
        arrays.append(np.frombuffer(body, dtype=np.float16, count=count // 2,
                                    offset=offset).reshape(desc["shape"]).copy())
        offset += count
    if offset != len(body):
        raise ValueError("Trailing bytes")
    return header["meta"], arrays
