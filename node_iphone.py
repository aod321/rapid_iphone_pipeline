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
import os
import signal
import socket
import struct
import threading
import time
from typing import Any, Dict, Optional

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

    HEALTH_CHECK_INTERVAL = 5
    HEALTH_CHECK_MAX_FAILURES = 3
    AUTO_RECOVERY_ENABLED = True
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
        self.service_info: Optional[ServiceInfo] = None

        # --- FastAPI ---
        self.app = FastAPI(title=f"{name} service (V2)")
        self._setup_routes()
        self._setup_lifecycle()

        # --- iPhone-specific state ---
        self._seq = 0
        self._client_connected = False
        self._last_frame_time = 0.0
        self._session_metadata: dict = {}

        # --- Signals ---
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ---- signal ----
    def _signal_handler(self, signum, frame):
        if self._shutdown_triggered:
            return
        self._shutdown_triggered = True
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

        @self.app.on_event("shutdown")
        async def _shutdown():
            self.logger.info("Shutdown signal received.")
            self.is_running = False
            await self._unregister_zeroconf()
            if hasattr(self, "loop_thread"):
                self.loop_thread.join(timeout=3)
            if hasattr(self, "_hb_thread"):
                self._hb_thread.join(timeout=2)
            try:
                self.data_pub.close(linger=0)
                self.zmq_ctx.term()
            except Exception:
                pass
            self.logger.info("Shutdown complete.")

    # ---- Zeroconf ----
    async def _register_zeroconf(self):
        self.zeroconf = AsyncZeroconf()
        local_ip = _get_local_ip()
        self.service_info = ServiceInfo(
            IPHONE_SERVICE_TYPE,
            f"{self.name}.{IPHONE_SERVICE_TYPE}",
            port=self.http_port,
            addresses=[socket.inet_aton(local_ip)],
            properties={
                b"data_port": str(self.data_port).encode(),
                b"tcp_port": str(self.tcp_port).encode(),
                b"node_name": self.name.encode(),
            },
        )
        await self.zeroconf.async_register_service(self.service_info)
        self.logger.info(f"Zeroconf registered: {self.name} @ {local_ip}:{self.http_port}")

    async def _unregister_zeroconf(self):
        if self.zeroconf and self.service_info:
            try:
                await self.zeroconf.async_unregister_service(self.service_info)
                await self.zeroconf.async_close()
            except Exception:
                pass
        self.zeroconf = None
        self.service_info = None

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
                if self._attempt_recovery(exc):
                    self.logger.info("Restarting main loop after recovery...")
                    continue
                else:
                    self.logger.error("Unrecoverable error, forcing exit")
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
        self.logger.info("Recovery successful (will restart TCP server)")
        return True

    # ---- iPhone-specific: hardware health ----
    def check_hardware_health(self):
        if not self._client_connected:
            raise RuntimeError("No iPhone connected")
        if time.time() - self._last_frame_time > 5.0:
            raise RuntimeError("iPhone data stale")

    # ---- iPhone-specific: main_loop runs asyncio TCP server ----
    def main_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_main())
        finally:
            loop.close()

    async def _async_main(self):
        server = await asyncio.start_server(
            self._handle_client, "0.0.0.0", self.tcp_port
        )
        self.logger.info(f"TCP server listening on 0.0.0.0:{self.tcp_port}")

        # mDNS: advertise TCP service so iPhone can discover us
        local_ip = _get_local_ip()
        tcp_service_info = ServiceInfo(
            "_vioserver._tcp.local.",
            f"VIO Server on {socket.gethostname()}._vioserver._tcp.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=self.tcp_port,
            properties={"hostname": socket.gethostname()},
        )
        aiozc = AsyncZeroconf()
        await aiozc.async_register_service(tcp_service_info)
        self.logger.info(f"[mDNS] Advertising _vioserver._tcp on {local_ip}:{self.tcp_port}")

        # Browse for iPhone apps
        listener = iPhoneServiceListener()
        browser = ServiceBrowser(aiozc.zeroconf, IPHONE_BROWSE_TYPE, listener)

        try:
            while self.is_running:
                await asyncio.sleep(1.0)
        finally:
            server.close()
            await server.wait_closed()
            browser.cancel()
            await aiozc.async_unregister_service(tcp_service_info)
            await aiozc.async_close()
            self.logger.info("TCP server and mDNS cleaned up")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        self.logger.info(f"Client connected: {addr}")
        self._client_connected = True
        self._last_frame_time = time.time()

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
                        "type": "iphone_meta",
                        "ts": time.time(),
                        **{k: v for k, v in meta.items()},
                    })

                elif msg_type == 1:
                    # Frame data
                    transform, device_ts, wall_clock, jpeg = decode_frame(payload)
                    self._seq += 1
                    self._last_frame_time = time.time()

                    self.publish({
                        "type": "iphone",
                        "seq": self._seq,
                        "ts": wall_clock,
                        "device_ts": device_ts,
                        "transform": payload[4:68],   # raw 64B transform bytes
                        "width": self._session_metadata.get("imageWidth", 0),
                        "height": self._session_metadata.get("imageHeight", 0),
                        "data": jpeg,
                    })

                    # Periodic logging
                    if self._seq % 100 == 0:
                        pos = transform[:3, 3]
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
        except Exception as exc:
            self.logger.error(f"Client handler error: {exc}")
        finally:
            self.logger.info(f"Client disconnected: {addr}")
            self._client_connected = False
            writer.close()
            try:
                await writer.wait_closed()
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
