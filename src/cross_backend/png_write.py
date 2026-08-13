"""Small RGB PNG writer for the isolated MuJoCo environment."""
import struct
import zlib
import numpy as np


def write_png(path, array):
    assert array.dtype == np.uint8 and array.ndim == 3 and array.shape[2] == 3
    height, width, _ = array.shape
    def chunk(name, data):
        return struct.pack('!I', len(data)) + name + data + struct.pack('!I', zlib.crc32(name + data) & 0xffffffff)
    rows = b''.join(b'\x00' + array[y].tobytes() for y in range(height))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows, 6)) + chunk(b'IEND', b''))
