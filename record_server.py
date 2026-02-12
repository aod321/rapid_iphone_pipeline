#!/usr/bin/env python3
"""Socket.IO server that receives iPhone ARKit VIO data and records trajectories.

Replaces the original iPhoneVIO/socketio_server.py with recording capability.
Saves both .npy (lossless) and .csv (TUM-compatible) on client disconnect.
"""

import argparse
import base64
import os
import struct
import time
from datetime import datetime

import eventlet
eventlet.hubs.use_hub('selects')
import numpy as np
import socketio
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Data decoding (from iPhoneVIO/socketio_server.py)
# ---------------------------------------------------------------------------

class DataPacket:
    def __init__(self, transform_matrix: np.ndarray, timestamp: float, wall_clock: float = 0.0):
        self.transform_matrix = transform_matrix.copy()
        self.timestamp = timestamp
        self.wall_clock = wall_clock

    def __str__(self):
        return (f"Translation: {self.transform_matrix[:3, 3]}, "
                f"Timestamp: {self.timestamp:.3f}, "
                f"WallClock: {self.wall_clock:.3f}")


def decode_data(encoded_str: str) -> DataPacket:
    """Decode base64-encoded ARKit frame (16 floats col-major + 1 double timestamp + 1 double wall_clock)."""
    data_bytes = base64.b64decode(encoded_str)
    transform_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            offset = 4 * (4 * i + j)
            transform_matrix[i, j] = struct.unpack('f', data_bytes[offset:offset + 4])[0]
    # Swift stores simd_float4x4 column-major; transpose to row-major
    transform_matrix = transform_matrix.T
    timestamp = struct.unpack('d', data_bytes[64:72])[0]
    wall_clock = struct.unpack('d', data_bytes[72:80])[0]
    return DataPacket(transform_matrix, timestamp, wall_clock)


# ---------------------------------------------------------------------------
# Per-session recording state
# ---------------------------------------------------------------------------

class SessionRecorder:
    def __init__(self, sid: str, output_dir: str):
        self.sid = sid
        self.output_dir = output_dir
        self.timestamps: list[float] = []
        self.wall_clocks: list[float] = []
        self.transforms: list[np.ndarray] = []
        self.prev_time: float = 0.0
        self.frame_count: int = 0

    def add_frame(self, packet: DataPacket):
        self.timestamps.append(packet.timestamp)
        self.wall_clocks.append(packet.wall_clock)
        self.transforms.append(packet.transform_matrix)
        self.frame_count += 1

        # Print live info
        fps = 1.0 / (packet.timestamp - self.prev_time) if self.prev_time > 0 else 0.0
        self.prev_time = packet.timestamp
        pos = packet.transform_matrix[:3, 3]
        print(f"[{self.sid[:8]}] frame={self.frame_count:5d}  "
              f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
              f"fps={fps:.1f}")

    def save(self):
        if self.frame_count == 0:
            print(f"[{self.sid[:8]}] No frames recorded, skipping save.")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.output_dir, f"trajectory_{tag}")

        timestamps = np.array(self.timestamps)
        wall_clocks = np.array(self.wall_clocks)
        transforms = np.stack(self.transforms)  # (N, 4, 4)

        # --- Save .npy ---
        npy_path = base + ".npy"
        np.save(npy_path, {"timestamps": timestamps, "wall_clocks": wall_clocks, "transforms": transforms})
        print(f"[{self.sid[:8]}] Saved {npy_path}  ({self.frame_count} frames)")

        # --- Save .csv (TUM-compatible: timestamp x y z qx qy qz qw + wall_clock) ---
        csv_path = base + ".csv"
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
        print(f"[{self.sid[:8]}] Saved {csv_path}")


# ---------------------------------------------------------------------------
# Socket.IO server
# ---------------------------------------------------------------------------

def create_server(output_dir: str):
    sio = socketio.Server()
    app = socketio.WSGIApp(sio)
    sessions: dict[str, SessionRecorder] = {}

    @sio.event
    def connect(sid, environ):
        print(f"\n=== Client connected: {sid} ===")
        sessions[sid] = SessionRecorder(sid, output_dir)

    @sio.event
    def disconnect(sid):
        print(f"\n=== Client disconnected: {sid} ===")
        recorder = sessions.pop(sid, None)
        if recorder:
            recorder.save()

    @sio.on('update')
    def handle_update(sid, data):
        recorder = sessions.get(sid)
        if recorder is None:
            return
        packet = decode_data(data)
        recorder.add_frame(packet)

    return app


def main():
    parser = argparse.ArgumentParser(description="iPhone VIO recording server")
    parser.add_argument("--port", type=int, default=5555, help="Listen port (default: 5555)")
    parser.add_argument("--output-dir", type=str, default="recordings",
                        help="Output directory for recordings (default: recordings)")
    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=True)
    app = create_server(args.output_dir)
    print(f"Recording server listening on 0.0.0.0:{args.port}")
    print(f"Recordings will be saved to: {os.path.abspath(args.output_dir)}/")
    eventlet.wsgi.server(eventlet.listen(('', args.port)), app)


if __name__ == '__main__':
    main()
