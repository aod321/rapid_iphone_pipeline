#!/usr/bin/env python3
"""TCP server that receives iPhone ARKit VIO data (pose + images) via raw binary protocol.

Replaces the Socket.IO-based record_server.py with a zero-dependency TCP approach.
Saves trajectory (.npy, .csv) and JPEG images on client disconnect.

Binary protocol:
  Header (8 bytes):
    [4B payload_len, uint32 LE]
    [1B msg_type]  — 0=session_metadata, 1=frame_data
    [3B reserved]

  Type 0 — Session Metadata: JSON blob (sent once on connect)
  Type 1 — Frame Data:
    [4B image_len, uint32 LE]
    [64B transform, float32 x 16, column-major]
    [8B device_timestamp, float64]
    [8B wall_clock, float64]
    [image_len B jpeg_data]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import struct
import time
from datetime import datetime
from typing import Optional

from zeroconf import ServiceInfo, ServiceBrowser, ServiceListener
from zeroconf.asyncio import AsyncZeroconf

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------

async def read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    """Read exactly n bytes from the stream, raising on EOF."""
    data = b""
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        data += chunk
    return data


def decode_transform(raw: bytes) -> np.ndarray:
    """Decode 64 bytes (16 x float32, column-major) into 4x4 row-major numpy array."""
    matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            offset = 4 * (4 * i + j)
            matrix[i, j] = struct.unpack_from('<f', raw, offset)[0]
    # Swift stores simd_float4x4 column-major; transpose to row-major
    return matrix.T


def decode_frame(payload: bytes):
    """Decode a type-1 frame payload.

    Returns: (transform_4x4, device_timestamp, wall_clock, jpeg_bytes)
    """
    image_len = struct.unpack_from('<I', payload, 0)[0]
    transform = decode_transform(payload[4:68])
    device_ts = struct.unpack_from('<d', payload, 68)[0]
    wall_clock = struct.unpack_from('<d', payload, 76)[0]
    jpeg_data = payload[84:84 + image_len]
    return transform, device_ts, wall_clock, jpeg_data


# ---------------------------------------------------------------------------
# Bonjour / mDNS helpers
# ---------------------------------------------------------------------------

def _get_local_ip() -> str:
    """Get the local IP address (best effort)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class iPhoneServiceListener(ServiceListener):
    """Log iPhone app presence via mDNS."""
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            addrs = [socket.inet_ntoa(a) for a in info.addresses]
            print(f"[mDNS] iPhone app online: {name} at {addrs}:{info.port}")
        else:
            print(f"[mDNS] iPhone app online: {name}")

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"[mDNS] iPhone app offline: {name}")

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Session recorder (pose + images)
# ---------------------------------------------------------------------------

class ImageSessionRecorder:
    def __init__(self, session_id: str, output_dir: str, metadata: Optional[dict] = None):
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = session_id[:8] if session_id else "unknown"
        self.session_dir = os.path.join(output_dir, f"session_{tag}_{short_id}")
        self.images_dir = os.path.join(self.session_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.session_id = session_id
        self.short_id = short_id
        self.metadata = metadata or {}
        self.timestamps: list[float] = []
        self.wall_clocks: list[float] = []
        self.transforms: list[np.ndarray] = []
        self.image_filenames: list[str] = []
        self.prev_time: float = 0.0
        self.frame_count: int = 0

        # Save metadata
        if metadata:
            meta_path = os.path.join(self.session_dir, "metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[{self.short_id}] Session dir: {self.session_dir}")
            print(f"[{self.short_id}] Metadata: {metadata.get('deviceModel', '?')} "
                  f"{metadata.get('imageWidth', '?')}x{metadata.get('imageHeight', '?')}")

    def add_frame(self, transform: np.ndarray, device_ts: float,
                  wall_clock: float, jpeg_data: bytes):
        filename = f"frame_{self.frame_count:06d}.jpg"

        self.timestamps.append(device_ts)
        self.wall_clocks.append(wall_clock)
        self.transforms.append(transform)
        self.image_filenames.append(filename)
        self.frame_count += 1

        # Print live info
        fps = 1.0 / (device_ts - self.prev_time) if self.prev_time > 0 else 0.0
        self.prev_time = device_ts
        pos = transform[:3, 3]
        jpeg_kb = len(jpeg_data) / 1024
        print(f"[{self.short_id}] frame={self.frame_count:5d}  "
              f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
              f"jpeg={jpeg_kb:.0f}KB  fps={fps:.1f}")

        if args_preview:
            img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                label = (f"pos:({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
                         f"fps:{fps:.0f}  #{self.frame_count}")
                cv2.putText(img, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                h, w = img.shape[:2]
                img = cv2.resize(img, (w // 2, h // 2))
                cv2.imshow("iPhone Preview", img)
                cv2.waitKey(1)

        # Return file path + data for async disk write
        return os.path.join(self.images_dir, filename), jpeg_data

    def save(self):
        if self.frame_count == 0:
            print(f"[{self.short_id}] No frames recorded, skipping save.")
            return

        timestamps = np.array(self.timestamps)
        wall_clocks = np.array(self.wall_clocks)
        transforms = np.stack(self.transforms)  # (N, 4, 4)

        # --- trajectory.npy (compatible with convert_trajectory.py) ---
        npy_path = os.path.join(self.session_dir, "trajectory.npy")
        np.save(npy_path, {
            "timestamps": timestamps,
            "wall_clocks": wall_clocks,
            "transforms": transforms,
        })
        print(f"[{self.short_id}] Saved {npy_path}  ({self.frame_count} frames)")

        # --- trajectory.csv (TUM format) ---
        csv_path = os.path.join(self.session_dir, "trajectory.csv")
        rotations = Rotation.from_matrix(transforms[:, :3, :3])
        quats = rotations.as_quat()  # (N, 4) as [qx, qy, qz, qw]
        positions = transforms[:, :3, 3]  # (N, 3)

        with open(csv_path, 'w') as f:
            f.write("timestamp,x,y,z,qx,qy,qz,qw,wall_clock\n")
            for i in range(self.frame_count):
                t = timestamps[i]
                wc = wall_clocks[i]
                x, y, z = positions[i]
                qx, qy, qz, qw = quats[i]
                f.write(f"{t:.6f},{x:.6f},{y:.6f},{z:.6f},"
                        f"{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f},{wc:.6f}\n")
        print(f"[{self.short_id}] Saved {csv_path}")

        # --- images.csv (frame index → timestamp → filename) ---
        img_csv_path = os.path.join(self.session_dir, "images.csv")
        with open(img_csv_path, 'w') as f:
            f.write("frame,timestamp,wall_clock,filename\n")
            for i in range(self.frame_count):
                f.write(f"{i},{timestamps[i]:.6f},{wall_clocks[i]:.6f},"
                        f"{self.image_filenames[i]}\n")
        print(f"[{self.short_id}] Saved {img_csv_path}")


# ---------------------------------------------------------------------------
# TCP server
# ---------------------------------------------------------------------------

def _write_file(filepath: str, data: bytes):
    """Write bytes to file (runs in thread pool)."""
    with open(filepath, 'wb') as f:
        f.write(data)


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    print(f"\n=== Client connected: {addr} ===")

    recorder: Optional[ImageSessionRecorder] = None
    loop = asyncio.get_event_loop()

    try:
        while True:
            # Read 8-byte header
            header = await read_exactly(reader, 8)
            payload_len = struct.unpack_from('<I', header, 0)[0]
            msg_type = header[4]

            # Read payload
            payload = await read_exactly(reader, payload_len)

            if msg_type == 0:
                # Session metadata
                metadata = json.loads(payload.decode('utf-8'))
                session_id = metadata.get("sessionId", "unknown")
                recorder = ImageSessionRecorder(
                    session_id=session_id,
                    output_dir=args_output_dir,
                    metadata=metadata,
                )

            elif msg_type == 1:
                # Frame data
                transform, device_ts, wall_clock, jpeg_data = decode_frame(payload)
                if recorder is None:
                    recorder = ImageSessionRecorder(
                        session_id="no_metadata",
                        output_dir=args_output_dir,
                    )
                filepath, data = recorder.add_frame(transform, device_ts, wall_clock, jpeg_data)
                # Disk write in thread pool — doesn't block event loop
                loop.run_in_executor(None, _write_file, filepath, data)

            else:
                print(f"Unknown message type: {msg_type}")

    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        print(f"\n=== Client disconnected: {addr} ===")
        if recorder:
            recorder.save()
        if args_preview:
            cv2.destroyAllWindows()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# Globals (set in main)
args_output_dir = "recordings"
args_preview = False


async def main_server(port: int, output_dir: str):
    global args_output_dir
    args_output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    server = await asyncio.start_server(handle_client, '0.0.0.0', port)
    print(f"TCP server listening on 0.0.0.0:{port}")
    print(f"Recordings will be saved to: {os.path.abspath(output_dir)}/")

    # --- Bonjour/mDNS advertising (async API to avoid EventLoopBlocked) ---
    hostname = socket.gethostname()
    local_ip = _get_local_ip()

    service_info = ServiceInfo(
        "_vioserver._tcp.local.",
        f"VIO Server on {hostname}._vioserver._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=port,
        properties={
            "hostname": hostname,
            "output_dir": os.path.basename(output_dir),
        },
    )

    aiozc = AsyncZeroconf()
    await aiozc.async_register_service(service_info)
    print(f"[mDNS] Advertising _vioserver._tcp on {local_ip}:{port}")

    # Browse for iPhone apps (ServiceBrowser works with the underlying Zeroconf)
    listener = iPhoneServiceListener()
    browser = ServiceBrowser(aiozc.zeroconf, "_iphonevio._tcp.local.", listener)

    try:
        async with server:
            await server.serve_forever()
    finally:
        print("[mDNS] Unregistering service...")
        await aiozc.async_unregister_service(service_info)
        await aiozc.async_close()


def main():
    parser = argparse.ArgumentParser(description="iPhone VIO TCP recording server (pose + images)")
    parser.add_argument("--port", type=int, default=5555,
                        help="Listen port (default: 5555)")
    parser.add_argument("--output-dir", type=str, default="recordings",
                        help="Output directory for recordings (default: recordings)")
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview window (requires OpenCV)")
    args = parser.parse_args()

    global args_preview
    args_preview = args.preview

    np.set_printoptions(precision=4, suppress=True)
    asyncio.run(main_server(args.port, args.output_dir))


if __name__ == '__main__':
    main()
