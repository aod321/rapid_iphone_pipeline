#!/usr/bin/env python3
"""node_iphone.py -- Self-contained iPhone ARKit VIO sensor node (V2 architecture).

Receives iPhone pose + image data via TCP binary protocol, publishes via ZMQ PUB
(msgpack serialized) for Recorder consumption. No dependency on zumi_pipeline.

Binary protocol (from iPhone app):
  Header (8 bytes): [4B payload_len uint32 LE] [1B msg_type] [3B reserved]
  Type 0 — Session Metadata: JSON blob (sent once on connect)
  Type 1 — Frame Data:
    [4B image_len uint32 LE] [64B transform float32x16 col-major]
    [8B device_timestamp float64] [8B wall_clock float64] [image_len B jpeg]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import socket
import struct
import threading
import time
from typing import Any, Dict, Optional, Set

import msgpack
import uvicorn
import zmq
from fastapi import FastAPI, HTTPException
from zeroconf import ServiceBrowser, ServiceInfo, ServiceListener
from zeroconf.asyncio import AsyncZeroconf

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
IPHONE_TCP_PORT = 5555
IPHONE_DATA_PORT = 5563          # ZMQ PUB port
IPHONE_HTTP_PORT = 8004
IPHONE_SERVICE_TYPE = "_zumi-iphone._tcp.local."
VIO_SERVICE_TYPE = "_vioserver._tcp.local."
IPHONE_BROWSE_TYPE = "_iphonevio._tcp.local."

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

# ---------------------------------------------------------------------------
# Protocol helpers (from tcp_server.py)
# ---------------------------------------------------------------------------

async def read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    """Read exactly *n* bytes, raising on premature EOF."""
    data = b""
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        data += chunk
    return data


def decode_transform(raw: bytes):
    """64 bytes (16 x float32, column-major) -> 4x4 row-major list-of-lists."""
    import numpy as np
    matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            offset = 4 * (4 * i + j)
            matrix[i, j] = struct.unpack_from('<f', raw, offset)[0]
    return matrix.T  # col-major -> row-major


def decode_frame(payload: bytes):
    """Decode a type-1 frame payload.

    Returns (transform_4x4, device_ts, wall_clock, jpeg_bytes).
    """
    image_len = struct.unpack_from('<I', payload, 0)[0]
    transform = decode_transform(payload[4:68])
    device_ts = struct.unpack_from('<d', payload, 68)[0]
    wall_clock = struct.unpack_from('<d', payload, 76)[0]
    jpeg_data = payload[84:84 + image_len]
    return transform, device_ts, wall_clock, jpeg_data


def split_timestamp(ts: float) -> tuple[int, int]:
    """Convert unix timestamp seconds to (sec, nsec)."""
    sec = int(ts)
    nsec = int((ts - sec) * 1e9)
    if nsec < 0:
        nsec = 0
    elif nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    return sec, nsec


def rotation_to_quaternion(transform) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> (x, y, z, w) quaternion."""
    r00, r01, r02 = float(transform[0, 0]), float(transform[0, 1]), float(transform[0, 2])
    r10, r11, r12 = float(transform[1, 0]), float(transform[1, 1]), float(transform[1, 2])
    r20, r21, r22 = float(transform[2, 0]), float(transform[2, 1]), float(transform[2, 2])

    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm > 1e-12:
        inv = 1.0 / norm
        x *= inv
        y *= inv
        z *= inv
        w *= inv
    else:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0

    return x, y, z, w


# ---------------------------------------------------------------------------
# mDNS helpers
# ---------------------------------------------------------------------------

def _get_local_ip() -> str:
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
    def add_service(self, zc, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            addrs = [socket.inet_ntoa(a) for a in info.addresses]
            logging.getLogger("mDNS").info(f"iPhone app online: {name} at {addrs}:{info.port}")
        else:
            logging.getLogger("mDNS").info(f"iPhone app online: {name}")

    def remove_service(self, zc, type_: str, name: str) -> None:
        logging.getLogger("mDNS").info(f"iPhone app offline: {name}")

    def update_service(self, zc, type_: str, name: str) -> None:
        pass


# ---------------------------------------------------------------------------
# IPhoneNode  (inlined StatelessSensorNode infrastructure)
# ---------------------------------------------------------------------------

class IPhoneNode:
    """Self-contained iPhone ARKit VIO sensor node.

    Inline the core StatelessSensorNode infra (ZMQ PUB, Zeroconf, FastAPI,
    health check, recovery) so this file has zero dependency on zumi_pipeline.
    """

    HEALTH_CHECK_INTERVAL = 2
    HEALTH_CHECK_MAX_FAILURES = 2
    # Fail fast and let external supervisor (rapid_driver/systemd) restart.
    AUTO_RECOVERY_ENABLED = False
    MAX_RECOVERY_ATTEMPTS = 10
    RECOVERY_BACKOFF_BASE = 2.0
    RECOVERY_BACKOFF_MAX = 60.0

    def __init__(
        self,
        name: str = "iphone_gp00",
        tcp_port: int = IPHONE_TCP_PORT,
        data_port: int = IPHONE_DATA_PORT,
        http_port: int = IPHONE_HTTP_PORT,
    ):
        self.name = name
        self.tcp_port = tcp_port
        self.data_port = data_port
        self.http_port = http_port

        self.is_running = False
        self.last_error: Optional[str] = None
        self._health_check_failures = 0
        self._shutdown_triggered = False
        self._recovery_attempts = 0
        self._in_recovery = False

        self.logger = logging.getLogger(self.name)

        # --- ZMQ PUB (msgpack) ---
        self.zmq_ctx = zmq.Context()
        self.data_pub = self.zmq_ctx.socket(zmq.PUB)
        self.data_pub.setsockopt(zmq.SNDHWM, 100)
        self.data_pub.bind(f"tcp://*:{data_port}")
        self.logger.info(f"ZMQ PUB bound to tcp://*:{data_port}")

        # --- Zeroconf ---
        self.zeroconf: Optional[AsyncZeroconf] = None
        self.http_service_info: Optional[ServiceInfo] = None
        self.vio_service_info: Optional[ServiceInfo] = None

        # --- FastAPI ---
        self.app = FastAPI(title=f"{name} service (V2)")
        self._setup_routes()
        self._setup_lifecycle()

        # --- iPhone-specific state ---
        self._seq = 0
        self._client_connected = False
        self._last_frame_time = 0.0
        self._session_metadata: dict = {}
        self._heartbeat_path = os.environ.get("RAPID_HEARTBEAT_PATH", "").strip() or None
        self._heartbeat_stop = threading.Event()
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._client_writers: Set[asyncio.StreamWriter] = set()
        self._client_writers_lock = threading.Lock()

        # --- Signals ---
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ---- signal ----
    def _signal_handler(self, signum, frame):
        if self._shutdown_triggered:
            return
        self._shutdown_triggered = True
        self.last_error = f"signal_{signum}"
        self._write_heartbeat()
        self.logger.info(f"Signal {signum} received, shutting down...")
        raise KeyboardInterrupt

    # ---- FastAPI routes ----
    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            try:
                self.check_hardware_health()
                return {"status": "ok", "node": self.name}
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Hardware check failed: {exc}")

        @self.app.get("/status")
        async def status():
            return self.status_payload()

    # ---- FastAPI lifecycle ----
    def _setup_lifecycle(self):
        @self.app.on_event("startup")
        async def _startup():
            self.logger.info("Initializing...")
            try:
                await self._register_zeroconf()
                self.is_running = True
            except Exception as exc:
                self.last_error = str(exc)
                self.logger.error(f"Init failed: {exc}")
                raise

            # main loop thread (runs asyncio TCP server)
            self.loop_thread = threading.Thread(target=self._main_loop_wrapper, daemon=True)
            self.loop_thread.start()

            # health check thread
            self._hb_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self._hb_thread.start()

            # heartbeat writer thread
            self._heartbeat_stop.clear()
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
            self._write_heartbeat()

        @self.app.on_event("shutdown")
        async def _shutdown():
            self.logger.info("Shutdown signal received.")
            self.is_running = False
            self._heartbeat_stop.set()
            self._request_client_shutdown()
            await self._unregister_zeroconf()
            if hasattr(self, "loop_thread"):
                self.loop_thread.join(timeout=5)
            if hasattr(self, "_hb_thread"):
                self._hb_thread.join(timeout=2)
            if hasattr(self, "_heartbeat_thread"):
                self._heartbeat_thread.join(timeout=2)
            try:
                self.data_pub.close(linger=0)
                self.zmq_ctx.term()
            except Exception:
                pass
            self._write_heartbeat()
            self.logger.info("Shutdown complete.")

    # ---- Zeroconf ----
    async def _register_zeroconf(self):
        self.zeroconf = AsyncZeroconf()
        local_ip = _get_local_ip()
        packed_ip = socket.inet_aton(local_ip)

        self.http_service_info = ServiceInfo(
            IPHONE_SERVICE_TYPE,
            f"{self.name}.{IPHONE_SERVICE_TYPE}",
            port=self.http_port,
            addresses=[packed_ip],
            properties={
                b"data_port": str(self.data_port).encode(),
                b"tcp_port": str(self.tcp_port).encode(),
                b"node_name": self.name.encode(),
            },
        )

        vio_service_name = f"VIO Server on {socket.gethostname()}"
        self.vio_service_info = ServiceInfo(
            VIO_SERVICE_TYPE,
            f"{vio_service_name}.{VIO_SERVICE_TYPE}",
            port=self.tcp_port,
            addresses=[packed_ip],
            properties={
                b"node_name": self.name.encode(),
                b"http_port": str(self.http_port).encode(),
                b"data_port": str(self.data_port).encode(),
            },
        )

        try:
            await self.zeroconf.async_register_service(self.http_service_info)
            await self.zeroconf.async_register_service(self.vio_service_info)
        except Exception:
            try:
                if self.vio_service_info:
                    await self.zeroconf.async_unregister_service(self.vio_service_info)
                if self.http_service_info:
                    await self.zeroconf.async_unregister_service(self.http_service_info)
            except Exception:
                pass
            await self.zeroconf.async_close()
            self.zeroconf = None
            self.http_service_info = None
            self.vio_service_info = None
            raise

        self.logger.info(
            "Zeroconf registered: "
            f"{self.name} API={local_ip}:{self.http_port} "
            f"TCP={local_ip}:{self.tcp_port}"
        )

    async def _unregister_zeroconf(self):
        if self.zeroconf:
            try:
                if self.vio_service_info:
                    await self.zeroconf.async_unregister_service(self.vio_service_info)
                if self.http_service_info:
                    await self.zeroconf.async_unregister_service(self.http_service_info)
                await self.zeroconf.async_close()
            except Exception:
                pass
        self.zeroconf = None
        self.http_service_info = None
        self.vio_service_info = None

    # ---- publish (msgpack over ZMQ) ----
    def publish(self, data: Dict[str, Any], topic: str = ""):
        if "ts" not in data:
            data["ts"] = time.time()
        if "node" not in data:
            data["node"] = self.name
        try:
            payload = msgpack.packb(data, use_bin_type=True)
            if topic:
                self.data_pub.send_multipart([topic.encode(), payload])
            else:
                self.data_pub.send(payload)
        except Exception as exc:
            self.logger.warning(f"Publish failed: {exc}")

    # ---- status ----
    def status_payload(self) -> Dict[str, Any]:
        payload = {
            "node": self.name,
            "is_running": self.is_running,
            "last_error": self.last_error,
            "ts": time.time(),
            "client_connected": self._client_connected,
            "frame_count": self._seq,
            "tcp_port": self.tcp_port,
            "session_metadata": {
                k: v for k, v in self._session_metadata.items()
                if not isinstance(v, bytes)
            } if self._session_metadata else {},
        }
        return payload

    def _heartbeat_payload(self) -> Dict[str, Any]:
        return {
            "updated_at": time.time(),
            "client_connected": self._client_connected,
            "last_frame_at": self._last_frame_time if self._last_frame_time > 0 else None,
            "seq": self._seq,
            "pid": os.getpid(),
            "error": self.last_error,
        }

    def _write_heartbeat(self) -> None:
        if not self._heartbeat_path:
            return
        tmp_path = None
        try:
            directory = os.path.dirname(self._heartbeat_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = (
                f"{self._heartbeat_path}.{os.getpid()}."
                f"{threading.get_ident()}.{time.time_ns()}.tmp"
            )
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._heartbeat_payload(), f, separators=(",", ":"))
            os.replace(tmp_path, self._heartbeat_path)
        except Exception as exc:
            self.logger.warning(f"Heartbeat write failed: {exc}")
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _heartbeat_loop(self) -> None:
        while self.is_running and not self._heartbeat_stop.is_set():
            self._write_heartbeat()
            self._heartbeat_stop.wait(0.5)

    # ---- start (blocking) ----
    def start(self):
        uvicorn.run(self.app, host="0.0.0.0", port=self.http_port, log_level="info")

    # ---- main loop wrapper with recovery ----
    def _main_loop_wrapper(self):
        while self.is_running:
            try:
                self.main_loop()
                break
            except Exception as exc:
                self.last_error = str(exc)
                self.logger.error(f"Main loop error: {exc}")
                self._write_heartbeat()
                if self._attempt_recovery(exc):
                    self.logger.info("Restarting main loop after recovery...")
                    continue
                else:
                    self.logger.error("Unrecoverable error, forcing exit")
                    self._write_heartbeat()
                    os._exit(1)

    # ---- health check loop ----
    def _health_check_loop(self):
        while self.is_running:
            if self._in_recovery:
                time.sleep(0.5)
                continue
            try:
                self.check_hardware_health()
                if self._health_check_failures > 0:
                    self.logger.info("Health check passed, resetting failure count")
                self._health_check_failures = 0
            except Exception as exc:
                self._health_check_failures += 1
                self.logger.warning(
                    f"Health check failed ({self._health_check_failures}/"
                    f"{self.HEALTH_CHECK_MAX_FAILURES}): {exc}"
                )
                if self._health_check_failures >= self.HEALTH_CHECK_MAX_FAILURES:
                    self.last_error = f"Hardware health check failed: {exc}"
                    self.logger.error(self.last_error)
                    self.logger.error("Health check threshold reached, exiting for supervisor restart")
                    self._write_heartbeat()
                    os._exit(1)
            time.sleep(self.HEALTH_CHECK_INTERVAL)

    # ---- recovery ----
    def _attempt_recovery(self, exc: Exception) -> bool:
        if not self.AUTO_RECOVERY_ENABLED:
            return False
        if self._recovery_attempts >= self.MAX_RECOVERY_ATTEMPTS:
            self.logger.error(f"Max recovery attempts ({self.MAX_RECOVERY_ATTEMPTS}) reached")
            return False
        self._in_recovery = True
        self._recovery_attempts += 1
        backoff = min(
            self.RECOVERY_BACKOFF_BASE * (2 ** (self._recovery_attempts - 1)),
            self.RECOVERY_BACKOFF_MAX,
        )
        self.logger.info(
            f"Recovery attempt {self._recovery_attempts}/{self.MAX_RECOVERY_ATTEMPTS}, "
            f"waiting {backoff:.1f}s..."
        )
        time.sleep(backoff)
        # For iPhone node, recovery just means restarting the TCP server loop
        self._client_connected = False
        self._recovery_attempts = 0
        self._health_check_failures = 0
        self._in_recovery = False
        self.last_error = None
        self._write_heartbeat()
        self.logger.info("Recovery successful (will restart TCP server)")
        return True

    # ---- iPhone-specific: hardware health ----
    def check_hardware_health(self):
        # iPhone hot-plug is normal. "not connected" is not a fatal error.
        if not self._client_connected:
            return
        if time.time() - self._last_frame_time > 5.0:
            raise RuntimeError("iPhone data stale")

    # ---- iPhone-specific: main_loop runs asyncio TCP server ----
    def main_loop(self):
        loop = asyncio.new_event_loop()
        self._main_loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_main())
        finally:
            # Drain any remaining callbacks before closing the loop
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            self._main_loop = None
            loop.close()

    def _request_client_shutdown(self):
        loop = self._main_loop
        if loop is None or not loop.is_running():
            return

        def _close_clients():
            with self._client_writers_lock:
                writers = list(self._client_writers)
            for writer in writers:
                try:
                    writer.close()
                except Exception:
                    pass

        try:
            loop.call_soon_threadsafe(_close_clients)
        except RuntimeError:
            pass

    async def _close_all_clients(self):
        with self._client_writers_lock:
            writers = list(self._client_writers)
        for writer in writers:
            try:
                writer.close()
            except Exception:
                pass
        for writer in writers:
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _async_main(self):
        server = await asyncio.start_server(
            self._handle_client, "0.0.0.0", self.tcp_port
        )
        self.logger.info(f"TCP server listening on 0.0.0.0:{self.tcp_port}")

        # Browse for iPhone apps (python zeroconf is fine for browsing — we use raw sockets)
        aiozc = AsyncZeroconf()
        listener = iPhoneServiceListener()
        browser = ServiceBrowser(aiozc.zeroconf, IPHONE_BROWSE_TYPE, listener)

        try:
            while self.is_running:
                await asyncio.sleep(1.0)
        finally:
            server.close()
            await server.wait_closed()
            await self._close_all_clients()
            browser.cancel()
            await aiozc.async_close()
            current = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            self.logger.info("TCP server and mDNS cleaned up")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        with self._client_writers_lock:
            self._client_writers.add(writer)
        self.logger.info(f"Client connected: {addr}")
        self._client_connected = True
        self.last_error = None
        self._last_frame_time = time.time()
        self._write_heartbeat()

        try:
            while self.is_running:
                # 8-byte header
                header = await read_exactly(reader, 8)
                payload_len = struct.unpack_from('<I', header, 0)[0]
                msg_type = header[4]

                payload = await read_exactly(reader, payload_len)

                if msg_type == 0:
                    # Session metadata (JSON)
                    meta = json.loads(payload.decode("utf-8"))
                    self._session_metadata = meta
                    self.logger.info(
                        f"Session metadata: {meta.get('deviceModel', '?')} "
                        f"{meta.get('imageWidth', '?')}x{meta.get('imageHeight', '?')} "
                        f"session={meta.get('sessionId', '?')[:8]}"
                    )
                    self.publish({
                        "type": "recording_start",
                        "ts": time.time(),
                        "session_id": meta.get("sessionId", ""),
                        **{k: v for k, v in meta.items()},
                    })

                elif msg_type == 1:
                    # Frame data
                    transform, _device_ts, wall_clock, jpeg = decode_frame(payload)
                    self._seq += 1
                    self._last_frame_time = time.time()
                    sec, nsec = split_timestamp(wall_clock)
                    qx, qy, qz, qw = rotation_to_quaternion(transform)
                    pos = transform[:3, 3]

                    self.publish({
                        "type": "image",
                        "ts": wall_clock,
                        "timestamp": {"sec": sec, "nsec": nsec},
                        "frame_id": "iphone_camera",
                        "format": "jpeg",
                        "data": jpeg,
                    })
                    self.publish({
                        "type": "iphone_pose",
                        "ts": wall_clock,
                        "timestamp": {"sec": sec, "nsec": nsec},
                        "frame_id": "world",
                        "pose": {
                            "position": {
                                "x": float(pos[0]),
                                "y": float(pos[1]),
                                "z": float(pos[2]),
                            },
                            "orientation": {
                                "x": qx,
                                "y": qy,
                                "z": qz,
                                "w": qw,
                            },
                        },
                    })

                    # Periodic logging
                    if self._seq % 100 == 0:
                        jpeg_kb = len(jpeg) / 1024
                        self.logger.info(
                            f"frame={self._seq:6d}  "
                            f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})  "
                            f"jpeg={jpeg_kb:.0f}KB"
                        )
                else:
                    self.logger.warning(f"Unknown msg_type={msg_type}")

        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.logger.error(f"Client handler error: {exc}")
            self.last_error = str(exc)
            self._write_heartbeat()
        finally:
            self.logger.info(f"Client disconnected: {addr}")
            self._client_connected = False
            self._write_heartbeat()
            with self._client_writers_lock:
                self._client_writers.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except (RuntimeError, OSError):
                pass
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="iPhone ARKit VIO sensor node (V2 architecture)"
    )
    parser.add_argument("--name", default="iphone_gp00", help="Node name (default: iphone_gp00)")
    parser.add_argument("--tcp-port", type=int, default=IPHONE_TCP_PORT, help="TCP listen port")
    parser.add_argument("--data-port", type=int, default=IPHONE_DATA_PORT, help="ZMQ PUB port")
    parser.add_argument("--http-port", type=int, default=IPHONE_HTTP_PORT, help="HTTP API port")
    args = parser.parse_args()

    node = IPhoneNode(
        name=args.name,
        tcp_port=args.tcp_port,
        data_port=args.data_port,
        http_port=args.http_port,
    )
    node.start()
