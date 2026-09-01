# Apple USB-NCM links

Exo can use the point-to-point IPv4 link-local network exposed when an Apple
Silicon Mac is connected to a Linux host over USB. Generic `169.254/16`
interfaces remain excluded from reachability probing.

Exo classifies a link as `apple_usb_ncm` only when the operating system proves
its identity:

- macOS reports the interface below `AppleUSBDeviceNCMData` in the I/O Registry.
- Linux binds the interface to `cdc_ncm` and reports USB vendor/product
  `05ac:1905` in its sysfs ancestry.

Linux needs a kernel containing upstream commit
`a5148bc2fa27092862ac4b9e7b5c8340d60cff34`, or an equivalent backport, for the
Apple device to bind successfully. The link also needs IPv4 addresses on the
same `169.254/16` subnet on both hosts.

`EXO_REACHABILITY_ALLOWED_CIDRS` continues to apply after classification. This
makes route-isolated testing possible: use the LAN subnet to force Ethernet or
`169.254.0.0/16` to force the verified USB-NCM path.

For model data, Exo ranks a measured wired Ethernet path ahead of Apple USB-NCM
and Apple USB-NCM ahead of Wi-Fi. Control-plane sockets continue to prefer a
broadly reachable LAN address.

On Linux, physical Ethernet classification requires an Ethernet-type sysfs
device with a bound driver; virtual bridges without a device driver stay
`unknown`. Placement fallbacks apply the same reachability filter as active
probes, so an excluded interface cannot silently re-enter route selection.
