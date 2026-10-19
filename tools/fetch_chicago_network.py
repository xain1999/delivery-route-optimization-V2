"""Download the Chicago street network used in Chapter 4.

    python tools/fetch_chicago_network.py        # about 32 MB, a few seconds

The network is the drivable street network of the city of Chicago from
OpenStreetMap, as an OSMnx GraphML file.  The book uses one fixed snapshot,
published in the cybergis/COVID-19AccessibilityNotebook repository, so that
every reader gets exactly the network behind the printed numbers.  To build
a current network instead, use tools/fetch_chicago_roads.py.

Map data (c) OpenStreetMap contributors, available under the Open Database
License (ODbL 1.0): https://www.openstreetmap.org/copyright
If you publish anything made from it, credit "(c) OpenStreetMap contributors".

Writes data/osm/Chicago_Network.graphml and checks its SHA-256.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/cybergis/COVID-19AccessibilityNotebook/"
       "9e0b9495fc15bce7f09d62c10a740f7087bb5863/data/Chicago_Network.graphml")
SHA256 = "37564a0f39b17beff7b85e8a59da920a8df833b9c36638ab11768f69be3bf5b5"
OUT = Path(__file__).resolve().parent.parent / "data" / "osm" / "Chicago_Network.graphml"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not (OUT.exists() and hashlib.sha256(OUT.read_bytes()).hexdigest() == SHA256):
        print("downloading the Chicago street network (about 32 MB)...", flush=True)
        with urllib.request.urlopen(URL, timeout=300) as resp:
            OUT.write_bytes(resp.read())
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if digest != SHA256:
        print(f"unexpected file (SHA-256 {digest}); delete {OUT} and retry", file=sys.stderr)
        return 1
    print(f"{OUT.name}: {OUT.stat().st_size / 1e6:.1f} MB, checksum OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
