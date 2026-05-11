"""Regression tests for the macOS ``networksetup -listallhardwareports`` parser.

The parser feeds ``find_ip_prioritised``'s ``type_priority`` axis, which then
picks the IP that the placement engine uses for the JACCL coordinator wire
and the asymmetric drafter wire. Two different bug classes have shown up
here and the test surface covers both:

1. **TB peer-cable vs TB-attached dock NIC.** macOS reports a TB-bridge
   member port (the cable peering two Macs over the 192.168.0.x /30
   subnet) as ``"Thunderbolt 1"`` / ``"Thunderbolt 2"`` / ``"Thunderbolt
   Bridge"`` -- without the substring ``"Ethernet"``. A TB *dock* with
   a built-in NIC (e.g. CalDigit / OWC / Apple Studio Display ethernet
   port) is reported as ``"Thunderbolt Ethernet Slot N"`` -- with
   ``"Ethernet"``. The two are functionally different: the peer cable is
   the high-bandwidth Mac-to-Mac link the placement engine wants for the
   data plane, while a TB-dock NIC is just a wired-LAN NIC that happens
   to connect via TB physical layer.

2. **USB-ethernet dongles on high-index enX.** Both smbp and smbpt run
   USB-LAN dongles whose hardware port is ``"USB 10/100/1G/2.5G/5G/10G
   LAN"`` on ``en16`` / ``en9``. These are dedicated wired ethernet
   adapters; ``find_ip_prioritised(ring=False)`` should pick them over
   Wi-Fi for the JACCL coordinator wire.

The first iteration of the fix special-cased ``startswith("Thunderbolt")``
*before* the ``Ethernet/LAN`` check and added a blanket
``enX -> maybe_ethernet`` downgrade. That correctly classified the peer
TB cables but had two regressions:

* ``"Thunderbolt Ethernet Slot N"`` was bucketed as ``thunderbolt``
  instead of ``ethernet``, mis-attributing the wired NIC behind a TB
  dock as a peer cable.
* The enX downgrade collapsed real ``"USB ... LAN"`` dongles to
  ``maybe_ethernet``, which deranks them under ``ring=False``
  priority and prevents wired LAN from winning the control-plane
  wire when no TB peer is reachable.

The current classifier keeps the order ``Wi-Fi -> Ethernet/LAN ->
Thunderbolt -> unknown`` and drops the device-name downgrade entirely.
The presence of ``"Ethernet"`` or ``"LAN"`` in the hardware-port name
is the authoritative wired-NIC signal, regardless of the underlying
physical bus.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from exo.utils.info_gatherer.system_info import (
    _get_interface_types_from_networksetup,  # pyright: ignore[reportPrivateUsage]
)


def _make_completed_process_stdout(text: str) -> object:
    """Stand-in for ``anyio.run_process``'s return value.

    ``_get_interface_types_from_networksetup`` only reads ``.stdout`` and
    decodes it; everything else on the return object is unused. A trivial
    namespace-like object with a ``stdout`` attribute is enough to drive
    the parser without depending on the full ``CompletedProcess`` shape.
    """

    class _Stub:
        def __init__(self, stdout_bytes: bytes) -> None:
            self.stdout = stdout_bytes

    return _Stub(text.encode())


# Captured verbatim from `wc-bmbp` (M5 Max MBP, macOS 26.5). The
# `Thunderbolt Ethernet Slot N` blocks here represent the dock NIC
# variant -- they MUST classify as `ethernet` (real wired LAN behind
# the TB-attached dock), not `thunderbolt`. The bare `Thunderbolt 1/2/3`
# and `Thunderbolt Bridge` blocks are the peer-cable case and stay
# `thunderbolt`.
_REAL_M5_MAX_NETWORKSETUP_OUTPUT = """\
Hardware Port: Wi-Fi
Device: en0
Ethernet Address: aa:bb:cc:dd:ee:00

Hardware Port: USB 10/100/1G/2.5G/5G/10G LAN
Device: en16
Ethernet Address: f4:4d:ad:08:5b:c6

Hardware Port: Ethernet Adapter (en3)
Device: en3
Ethernet Address: 0e:cb:fb:d8:b1:10

Hardware Port: Thunderbolt Ethernet Slot 0
Device: en10
Ethernet Address: 64:4b:f0:80:05:84

Hardware Port: Thunderbolt Ethernet Slot 1, Port 1
Device: en11
Ethernet Address: 64:4b:f0:80:05:85

Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: 36:79:2c:66:09:80

Hardware Port: Thunderbolt 1
Device: en1
Ethernet Address: 36:79:2c:66:09:80

Hardware Port: Thunderbolt 2
Device: en6
Ethernet Address: 36:79:2c:66:09:84

Hardware Port: Thunderbolt 3
Device: en2
Ethernet Address: 36:79:2c:66:09:88
"""


@pytest.mark.anyio
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="networksetup parser only runs on macOS",
)
async def test_thunderbolt_dock_nic_classified_as_ethernet() -> None:
    """``Thunderbolt Ethernet Slot N`` is a dock NIC -> wired ethernet.

    A TB-attached dock with a built-in ethernet port (CalDigit, OWC,
    etc.) is reported by macOS with this hardware-port name. The NIC
    is a real wired LAN adapter that just happens to be electrically
    plumbed in over a TB cable -- it carries 1 GbE / 2.5 GbE / 10 GbE
    LAN traffic, not Mac-to-Mac TB-bridge traffic. It must classify
    as ``ethernet`` so the JACCL coordinator wire can prefer it over
    Wi-Fi.
    """
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        return_value=_make_completed_process_stdout(_REAL_M5_MAX_NETWORKSETUP_OUTPUT),
    ):
        types = await _get_interface_types_from_networksetup()

    assert types["en10"] == "ethernet"
    assert types["en11"] == "ethernet"


@pytest.mark.anyio
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="networksetup parser only runs on macOS",
)
async def test_thunderbolt_peer_cable_classified_as_thunderbolt() -> None:
    """``Thunderbolt N`` (no ``Ethernet`` substring) is the peer cable.

    macOS labels the actual TB-bridge member ports -- the ones with
    the 192.168.0.x /30 subnet between two Macs -- as plain
    ``"Thunderbolt 1"`` / ``"Thunderbolt 2"`` / ``"Thunderbolt 3"`` /
    ``"Thunderbolt Bridge"``. These must stay ``thunderbolt`` so the
    ring data plane and the v3 socket-only drafter wire can prefer
    them when topology has them registered.
    """
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        return_value=_make_completed_process_stdout(_REAL_M5_MAX_NETWORKSETUP_OUTPUT),
    ):
        types = await _get_interface_types_from_networksetup()

    assert types["en1"] == "thunderbolt"
    assert types["en2"] == "thunderbolt"
    assert types["en6"] == "thunderbolt"
    assert types["bridge0"] == "thunderbolt"


@pytest.mark.anyio
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="networksetup parser only runs on macOS",
)
async def test_usb_lan_dongle_classified_as_ethernet() -> None:
    """USB-ethernet dongles on enX must classify as ``ethernet``.

    smbp / smbpt both run USB-LAN dongles whose macOS hardware-port
    name is ``"USB 10/100/1G/2.5G/5G/10G LAN"``. These are dedicated
    wired NICs -- typically 2.5 / 5 / 10 GbE depending on the dongle
    -- and the JACCL coordinator selection ranks them strictly above
    Wi-Fi. An earlier iteration of this parser unconditionally
    downgraded any ``enX`` device (other than ``en0`` / ``en1``) to
    ``maybe_ethernet`` based on the device name alone, which masked
    the dongle's true type and forced the coordinator wire onto an
    asymmetric-routed path. The downgrade was dropped; the
    hardware-port name is now authoritative.
    """
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        return_value=_make_completed_process_stdout(_REAL_M5_MAX_NETWORKSETUP_OUTPUT),
    ):
        types = await _get_interface_types_from_networksetup()

    assert types["en16"] == "ethernet"


@pytest.mark.anyio
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="networksetup parser only runs on macOS",
)
async def test_ethernet_adapter_stub_classified_as_ethernet() -> None:
    """``Ethernet Adapter (enX)`` stubs classify as ``ethernet``.

    These are macOS-internal interfaces (often AF_LINK / virtio /
    AppleAVB) that have no usable IPv4 in normal operation; they get
    filtered out one layer up in ``get_network_interfaces`` because
    psutil emits them without AF_INET addresses. The classification
    only matters in the corner case where one happens to carry a
    link-local IPv4 (``169.254.x.x``); even then they lose the
    ``_address_priority`` sort to any real ``192.168.x.x`` candidate.

    No behavioural concern -- just locking the classification in so
    later refactors don't accidentally change it.
    """
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        return_value=_make_completed_process_stdout(_REAL_M5_MAX_NETWORKSETUP_OUTPUT),
    ):
        types = await _get_interface_types_from_networksetup()

    assert types["en3"] == "ethernet"


@pytest.mark.anyio
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="networksetup parser only runs on macOS",
)
async def test_wifi_classified_as_wifi() -> None:
    """The Wi-Fi branch is unchanged but worth covering once.

    Ensures the reorder didn't accidentally drop the Wi-Fi check or let
    the more general ``Ethernet`` substring leak past it.
    """
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        return_value=_make_completed_process_stdout(_REAL_M5_MAX_NETWORKSETUP_OUTPUT),
    ):
        types = await _get_interface_types_from_networksetup()

    assert types["en0"] == "wifi"
