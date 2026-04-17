"""
mDNS/Bonjour service discovery for the BitResearch swarm.
Uses Zeroconf (Python implementation of mDNS/DNS-SD) to:
  - Advertise coordinator presence on local network
  - Discover workers that join the swarm
  - Advertise worker availability
  - Discover coordinator for workers to connect to
"""

import json
import logging
import socket
import time
from typing import Callable

from zeroconf import ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_bitresearch._tcp.local."
COORDINATOR_NAME = "bitresearch-coordinator"
WORKER_PREFIX = "bitresearch-worker-"


def get_local_ip() -> str:
    """Get the primary local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class CoordinatorAdvertiser:
    """Advertise the coordinator service on the local network via mDNS."""

    def __init__(self, port: int, metadata: dict | None = None):
        self.port = port
        self.metadata = metadata or {}
        self.zeroconf = None
        self.service_info = None

    def start(self):
        """Start advertising."""
        self.zeroconf = Zeroconf()
        ip = get_local_ip()

        properties = {
            "version": "0.1.0",
            "role": "coordinator",
        }
        properties.update({k: str(v) for k, v in self.metadata.items()})

        self.service_info = ServiceInfo(
            SERVICE_TYPE,
            f"{COORDINATOR_NAME}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=self.port,
            properties=properties,
        )
        self.zeroconf.register_service(self.service_info)
        logger.info(f"Coordinator advertised at {ip}:{self.port}")

    def stop(self):
        """Stop advertising."""
        if self.zeroconf and self.service_info:
            self.zeroconf.unregister_service(self.service_info)
            self.zeroconf.close()
            logger.info("Coordinator advertisement stopped")


class WorkerAdvertiser:
    """Advertise a worker node on the local network via mDNS."""

    def __init__(self, port: int, worker_id: str, hardware_info: dict | None = None):
        self.port = port
        self.worker_id = worker_id
        self.hardware_info = hardware_info or {}
        self.zeroconf = None
        self.service_info = None

    def start(self):
        """Start advertising."""
        self.zeroconf = Zeroconf()
        ip = get_local_ip()

        properties = {
            "version": "0.1.0",
            "role": "worker",
            "worker_id": self.worker_id,
            "tier": self.hardware_info.get("tier", "unknown"),
            "chip": self.hardware_info.get("chip", "unknown"),
            "memory_gb": str(self.hardware_info.get("total_memory_gb", 0)),
        }

        service_name = f"{WORKER_PREFIX}{self.worker_id}"
        self.service_info = ServiceInfo(
            SERVICE_TYPE,
            f"{service_name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=self.port,
            properties=properties,
        )
        self.zeroconf.register_service(self.service_info)
        logger.info(f"Worker {self.worker_id} advertised at {ip}:{self.port}")

    def stop(self):
        """Stop advertising."""
        if self.zeroconf and self.service_info:
            self.zeroconf.unregister_service(self.service_info)
            self.zeroconf.close()
            logger.info(f"Worker {self.worker_id} advertisement stopped")


class ServiceDiscovery:
    """Discover BitResearch services on the local network."""

    def __init__(self, on_found: Callable | None = None, on_removed: Callable | None = None):
        self.on_found = on_found
        self.on_removed = on_removed
        self.services: dict[str, dict] = {}
        self.zeroconf = None
        self.browser = None

    def _on_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ):
        if state_change == ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
                service_data = {
                    "name": name,
                    "addresses": addresses,
                    "port": info.port,
                    "properties": {
                        k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                        for k, v in info.properties.items()
                    },
                }
                self.services[name] = service_data
                logger.info(f"Discovered: {name} at {addresses}:{info.port}")
                if self.on_found:
                    self.on_found(service_data)

        elif state_change == ServiceStateChange.Removed:
            if name in self.services:
                removed = self.services.pop(name)
                logger.info(f"Removed: {name}")
                if self.on_removed:
                    self.on_removed(removed)

    def start(self):
        """Start discovering services."""
        self.zeroconf = Zeroconf()
        self.browser = ServiceBrowser(
            self.zeroconf,
            SERVICE_TYPE,
            handlers=[self._on_service_state_change],
        )
        logger.info("Service discovery started")

    def stop(self):
        """Stop discovering."""
        if self.browser:
            self.browser.cancel()
        if self.zeroconf:
            self.zeroconf.close()
        logger.info("Service discovery stopped")

    def find_coordinator(self, timeout: float = 10.0) -> dict | None:
        """Block until coordinator is found or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            for name, data in self.services.items():
                props = data.get("properties", {})
                if props.get("role") == "coordinator":
                    return data
            time.sleep(0.5)
        return None

    def find_workers(self) -> list[dict]:
        """Return all currently known workers."""
        workers = []
        for name, data in self.services.items():
            props = data.get("properties", {})
            if props.get("role") == "worker":
                workers.append(data)
        return workers
