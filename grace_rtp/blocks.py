"""Independent DEFLATE entropy streams; no upstream codec code is distributed."""
import math
import struct
import zlib
import numpy as np
from aiortc.rtp import RtpPacket

SSRC = 0x47524143
PT = 110
# version, frame, block, fragment, fragment count, complete block CRC32
HEADER = struct.Struct("!BIBHHI")


def compress_features(code, count=10):
    """Interleave symbols so every block covers motion AND residual features."""
    code = np.asarray(code, dtype="<f4").reshape(-1)
    return [zlib.compress(code[b::count].tobytes(), 6) for b in range(count)]


def restore_features(blocks, length, count=10):
    out = np.zeros(length, dtype="<f4")
    for b, data in enumerate(blocks):
        if data is None:
            continue
        raw = zlib.decompress(data)
        values = np.frombuffer(raw, dtype="<f4")
        if len(values) != len(out[b::count]):
            raise ValueError("Feature block length mismatch")
        out[b::count] = values
    return out


def packetize(blocks, frame_id, start_seq=0, mtu=1200, fps=30):
    """mtu is max RTP bytes. IPv4/UDP adds 28 bytes; SRTP adds its tag."""
    capacity = mtu - 12 - HEADER.size
    if capacity <= 0:
        raise ValueError("RTP MTU too small")
    result = []
    for block_id, block in enumerate(blocks):
        crc = zlib.crc32(block)
        n = max(1, math.ceil(len(block) / capacity))
        if n > 65535:
            raise ValueError("Too many block fragments")
        for fragment in range(n):
            payload = HEADER.pack(1, frame_id, block_id, fragment, n, crc)
            payload += block[fragment * capacity:(fragment + 1) * capacity]
            packet = RtpPacket(payload_type=PT, sequence_number=(start_seq + len(result)) % 65536,
                               timestamp=round(frame_id * 90000 / fps) % 2**32, ssrc=SSRC,
                               marker=int(block_id == len(blocks)-1 and fragment == n-1), payload=payload)
            result.append(packet.serialize())
    return result


class BlockAssembler:
    """One frame per instance. Reliable manifest permits all-packet-lost detection."""
    def __init__(self, frame_id, blocks=10):
        self.frame_id = frame_id
        self.count = blocks
        self.fragments = {}
        self.invalid = set()

    def add(self, data):
        packet = RtpPacket.parse(data)
        if packet.ssrc != SSRC or packet.payload_type != PT:
            raise ValueError("Wrong stream")
        if len(packet.payload) < HEADER.size:
            raise ValueError("Truncated block header")
        v, frame, b, i, n, crc = HEADER.unpack_from(packet.payload)
        if v != 1 or frame != self.frame_id or b >= self.count or n == 0 or i >= n:
            raise ValueError("Invalid block header")
        old_n, old_crc, parts = self.fragments.setdefault(b, (n, crc, {}))
        body = packet.payload[HEADER.size:]
        if (n, crc) != (old_n, old_crc) or (i in parts and parts[i] != body):
            self.invalid.add(b)
        parts[i] = body

    def finish(self):
        result = [None] * self.count
        for b, (n, crc, parts) in self.fragments.items():
            if b in self.invalid or len(parts) != n:
                continue
            data = b"".join(parts[i] for i in range(n))
            if zlib.crc32(data) == crc:
                result[b] = data
        return result
