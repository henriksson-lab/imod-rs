#!/usr/bin/env python3
"""Drop manifest entries that no test looks up any more.

Usage: prune-manifest.py <access-log> <manifest>...

The access log comes from running the suites that use the manifests with
IMOD_RS_GOLDEN_ACCESS_LOG=<access-log> (tests/common/golden.rs appends
"<manifest>\t<key>" for every golden looked up).  Each named manifest is
rewritten keeping only the entries whose key appears in the log for it --
what is left behind when cases are pruned from cases.tsv (fixtures/README.md,
"Pruned cases").  Run every test that reads a manifest before pruning it.
"""
import os
import sys


def parse(data):
    entries, at = [], 0
    header = []
    while at < len(data):
        end = data.index(b'\n', at)
        line = data[at:end]
        at = end + 1
        if line.startswith(b'#'):
            header.append(line + b'\n')
            continue
        fields = line.split(b' ')
        if fields[0] == b'text':
            size = int(fields[1])
            key = b' '.join(fields[2:]).decode()
            entries.append((key, line + b'\n' + data[at:at + size + 1]))
            at += size + 1
        else:
            entries.append((b' '.join(fields[3:]).decode(), line + b'\n'))
    return header, entries


def main():
    log, manifests = sys.argv[1], sys.argv[2:]
    used = {}
    for line in open(log):
        manifest, key = line.rstrip('\n').split('\t', 1)
        used.setdefault(os.path.realpath(manifest), set()).add(key)
    for manifest in manifests:
        keep = used.get(os.path.realpath(manifest), set())
        header, entries = parse(open(manifest, 'rb').read())
        kept = [blob for key, blob in entries if key in keep]
        with open(manifest, 'wb') as f:
            f.write(b''.join(header) + b''.join(kept))
        print('%s: kept %d of %d entries' % (manifest, len(kept), len(entries)))


main()
