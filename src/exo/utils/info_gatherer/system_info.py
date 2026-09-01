import platform
import re
import socket
import sys
from pathlib import Path
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.profiling import InterfaceType, NetworkInterfaceInfo

_APPLE_USB_NCM_IOREG_CLASS = "AppleUSBDeviceNCMData"
_APPLE_USB_VENDOR_ID = "05ac"
_APPLE_USB_NCM_PRODUCT_ID = "1905"
_LINUX_SYS_CLASS_NET = Path("/sys/class/net")


def get_os_version() -> str:
    """Return the OS version string for this node.

    On macOS this is the macOS version (e.g. ``"15.3"``).
    On other platforms it falls back to the platform name (e.g. ``"Linux"``).
    """
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        return version if version else "Unknown"
    return platform.system() or "Unknown"


async def get_os_build_version() -> str:
    """Return the macOS build version string (e.g. ``"24D5055b"``).

    On non-macOS platforms, returns ``"Unknown"``.
    """
    if sys.platform != "darwin":
        return "Unknown"

    try:
        process = await run_process(["sw_vers", "-buildVersion"])
    except CalledProcessError:
        return "Unknown"

    return process.stdout.decode("utf-8", errors="replace").strip() or "Unknown"


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    if sys.platform != "darwin":
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


async def _get_interface_types_from_networksetup() -> dict[str, InterfaceType]:
    """Parse networksetup -listallhardwareports to get interface types."""
    if sys.platform != "darwin":
        return {}

    try:
        result = await run_process(["networksetup", "-listallhardwareports"])
    except CalledProcessError:
        return {}

    types: dict[str, InterfaceType] = {}
    current_type: InterfaceType = "unknown"

    for line in result.stdout.decode().splitlines():
        if line.startswith("Hardware Port:"):
            port_name = line.split(":", 1)[1].strip()
            # Classification by hardware-port name. Order matters because
            # several macOS port names overlap in keywords.
            #
            # Examples observed on the team-wcv cluster:
            #
            #   "Wi-Fi"                                  -> wifi
            #   "USB 10/100/1G/2.5G/5G/10G LAN"          -> ethernet (dongle)
            #   "Thunderbolt Ethernet Slot 1, Port 1"    -> ethernet (TB
            #       dock NIC: a wired NIC sitting BEHIND a TB-attached
            #       dock; carries real wired LAN, NOT a peer-cable TB
            #       bridge)
            #   "Ethernet Adapter (en3)"                 -> ethernet
            #       (macOS-internal stubs; later filtered out because
            #       they have no usable IPv4)
            #   "Thunderbolt 1" / "Thunderbolt 2" / "Thunderbolt 3"
            #   "Thunderbolt Bridge"                     -> thunderbolt
            #       (peer-to-peer TB bridge cable between two Macs,
            #       carrying the 192.168.0.x /30 subnets)
            #
            # The presence of "Ethernet" or "LAN" in the hardware-port
            # name is the authoritative signal that an actual wired NIC
            # sits behind the port -- regardless of whether the NIC is
            # plumbed in via USB, PCIe, or a TB dock. The earlier
            # "Thunderbolt prefix wins" rule was wrong: it swept
            # TB-dock NICs into the thunderbolt bucket, which deranks
            # them under MlxJaccl coordinator selection (where
            # ring=False puts thunderbolt LAST) and prevents real
            # wired LAN from winning the control-plane wire.
            if "Wi-Fi" in port_name:
                current_type = "wifi"
            elif "Ethernet" in port_name or "LAN" in port_name:
                current_type = "ethernet"
            elif port_name.startswith("Thunderbolt"):
                current_type = "thunderbolt"
            else:
                current_type = "unknown"
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            # The hardware-port name is authoritative; no device-name
            # downgrade. Macs label TB peer-cable ports as "Thunderbolt N"
            # without "Ethernet" in them, so the port-name pass above
            # already distinguishes those from the TB dock NICs whose
            # names contain "Ethernet".
            types[device] = current_type

    return types


async def _get_apple_usb_ncm_interfaces_from_ioreg() -> set[str]:
    """Return macOS interfaces backed by Apple's USB-device NCM driver."""
    if sys.platform != "darwin":
        return set()

    try:
        result = await run_process(
            ["ioreg", "-r", "-c", _APPLE_USB_NCM_IOREG_CLASS, "-l", "-w0"]
        )
    except CalledProcessError:
        return set()

    output = result.stdout.decode("utf-8", errors="replace")
    return set(re.findall(r'"BSD Name"\s*=\s*"([^"\s]+)"', output))


def _read_sysfs_value(path: Path) -> str | None:
    try:
        return path.read_text().strip().lower()
    except OSError:
        return None


def _get_apple_usb_ncm_interfaces_from_sysfs(
    sys_class_net: Path = _LINUX_SYS_CLASS_NET,
) -> set[str]:
    """Return Linux interfaces matching cdc_ncm on Apple USB 05ac:1905."""
    try:
        interfaces = list(sys_class_net.iterdir())
    except OSError:
        return set()

    matched: set[str] = set()
    for interface in interfaces:
        try:
            device = (interface / "device").resolve(strict=True)
            driver = (device / "driver").resolve(strict=True).name
        except OSError:
            continue
        if driver != "cdc_ncm":
            continue

        for ancestor in (device, *device.parents):
            if (
                _read_sysfs_value(ancestor / "idVendor") == _APPLE_USB_VENDOR_ID
                and _read_sysfs_value(ancestor / "idProduct")
                == _APPLE_USB_NCM_PRODUCT_ID
            ):
                matched.add(interface.name)
                break

    return matched


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    interface_types = await _get_interface_types_from_networksetup()
    if sys.platform == "darwin":
        apple_usb_ncm_interfaces = await _get_apple_usb_ncm_interfaces_from_ioreg()
    elif sys.platform == "linux":
        apple_usb_ncm_interfaces = _get_apple_usb_ncm_interfaces_from_sysfs()
    else:
        apple_usb_ncm_interfaces = set()

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interface_type: InterfaceType = interface_types.get(
                        iface, "unknown"
                    )
                    if iface in apple_usb_ncm_interfaces:
                        interface_type = "apple_usb_ncm"
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=interface_type,
                        )
                    )
                case _:
                    pass

    return interfaces_info


async def get_model_and_chip() -> tuple[str, str]:
    """Get Mac system information using system_profiler."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    # TODO: better non mac support
    if sys.platform != "darwin":
        return (model, chip)

    try:
        process = await run_process(
            [
                "system_profiler",
                "SPHardwareDataType",
            ]
        )
    except CalledProcessError:
        return (model, chip)

    # less interested in errors here because this value should be hard coded
    output = process.stdout.decode().strip()

    model_line = next(
        (line for line in output.split("\n") if "Model Name" in line), None
    )
    model = model_line.split(": ")[1] if model_line else "Unknown Model"

    chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
    chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

    return (model, chip)
