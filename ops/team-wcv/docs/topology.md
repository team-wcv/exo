# Thunderbolt topology — 2-node wiring

JACCL's data path is RDMA over Thunderbolt. For 2 nodes the topology is
trivial: **one direct Thunderbolt 5 cable** between Bus 2 of each Mac.
The normal day-to-day shape is `wc-smbp + wc-smbpt`, or `wc-smbp +
wc-bmbp` when the operator Mac is also the second node.

```
+------------------+                           +-------------------+
|     wc-smbp      |                           |     wc-smbpt      |
|  (rank0, M5 Max) |                           |  (rank1, M-series)|
|                  |                           |                   |
|  TB5 Bus 1 ------|--> dock / display chain   | TB5 Bus 1 -- free |
|  TB5 Bus 2 ===== | <======= TB5 cable =====> | TB5 Bus 2 ======  |
+--------|---------+                           +---------|---------+
         |                                               |
         +--------------- LAN (Ethernet) ----------------+
                            |
                            v
                    rank0 TCP coordinator /
                    Thunderbolt control IP
```

## Rules

1. **Use Bus 2 on each Mac** for the RDMA cable. Bus 1 is usually
   already driving a dock or display, which forces shared bandwidth and
   can degrade RDMA throughput.
2. **Direct cable, no dock in the middle.** A dock between the two Macs
   downgrades the link and breaks RDMA.
3. **Cable must be Thunderbolt 5 (or at minimum TB4 / 40 Gb/s).**
   Apple's USB-C charge cable is electrically a USB 2.0 cable in
   disguise; it will silently negotiate down to 480 Mb/s.
4. **Do not confuse the cluster control path with the client API URL.**
   `config/cluster.json` should use the worker-reachable control IPs for
   the fabric you are actually clustering over. Your OpenAI client may use
   a different operator-reachable URL.

## Bridge Mode vs RDMA Mode

Thunderbolt Bridge and JACCL RDMA are useful at different times:

- Bridge mode is useful for normal IP networking over the Thunderbolt cable,
  including `ssh` and `rsync` model copies.
- RDMA mode is required for JACCL runtime traffic. In this mode, `bridge0`
  must not own the Thunderbolt ports.

The intended operator flow is:

```bash
bb rdma bridge-on all
bb model sync-bridge <model>
bb rdma bridge-off all
bb rdma bridge-status all
bb use <model>
```

For the combined flow:

```bash
bb use-bridge-sync <model>
```

Both `bridge-on` and `bridge-off` require sudo because they change macOS
network services. The bridge snapshot is stored in `state/rdma-bridge/`.

These `bb` commands are historical Big Brain helper commands. Keep the topology
rules, but do not rely on this CLI unless it has been intentionally migrated into
the current EXO ops tree.

## Verifying the link

```bash
bb topology check
bb topology discover --json
```

Look for, on each side:

```
Thunderbolt/USB4 Bus 2:
  Status:        Device connected
  Link Status:   0x2
  Speed:         80 Gb/s     # TB4
  Device Name:   MacBook Pro # the peer
```

For TB5 you should see `Speed: 120 Gb/s`.

If `Status: No device connected`, the cable isn't seated or the bus is
disabled. If `Speed: 40 Gb/s` instead of 80/120, the cable isn't TB-rated.

## Naming the RDMA device

After Recovery `rdma_ctl enable` + reboot, run on each Mac:

```bash
ibv_devices
```

You'll see something like:

```
device          	   node GUID
------          	----------------
rdma_en3        	0000000005ac0001
```

The `rdma_enN` name goes into `config/cluster.json`'s `rdma` matrix:

```json
[
  { "ssh": "wc-smbp",  "ips": ["10.254.0.2"], "rdma": [null,         "rdma_en1"] },
  { "ssh": "wc-bmbp",  "ips": ["10.254.0.1"], "rdma": ["rdma_en6",   null      ] }
]
```

Do not assume both sides use the same `rdma_enN`. The safe mapping
procedure is:

1. Confirm the inter-node control IPs route over `bridge0`.
2. On each Mac, identify the active Thunderbolt IP port:
   `ioreg -r -c AppleThunderboltIPPort -l | egrep 'AppleThunderboltIPPort|BSD Name'`
3. Map the active Thunderbolt `en*` to the RDMA interface for that Mac.
4. Put the per-node result into `config/cluster.json`:
   entry `[i][j]` is the RDMA device on node `i` that talks to node `j`.

Real example from the current 2-node `wc-smbp + wc-bmbp` setup:

- `wc-smbp`: active Thunderbolt IP port is `en1`, so the peer-facing RDMA device is `rdma_en1`
- `wc-bmbp`: active Thunderbolt IP port is `en6`, so the peer-facing RDMA device is `rdma_en6`

This is why `ibv_devices` alone is not enough: both Macs may list several
`rdma_en*` devices, but only one is backed by the active Thunderbolt link.

`bb topology discover` automates most of this inspection. It reports
Thunderbolt IP ports, Bridge membership, RDMA devices, bus status, link
speed, and a recommended config. Use `--write-config state/discovered.json`
to save the recommendation without overwriting the active local
`config/cluster.json`.
