"""Download only selected scLDM release artifacts (the repo's downloader fetches whole groups).

Example (Replogle negative-binomial checkpoint + config + size factors):
    python experiments/perturbation_direction/fetch_artifacts.py --match replogle --exclude gaussian
"""

from __future__ import annotations

import argparse
import pathlib

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from scldm.download_artifacts import ARTIFACTS, DEFAULT_DESTINATION, _download_file, _parse_s3_uri


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", nargs="+", required=True, help="keep artifacts whose path contains any of these")
    ap.add_argument("--exclude", nargs="*", default=[], help="drop artifacts whose path contains any of these")
    ap.add_argument("--destination", default=str(DEFAULT_DESTINATION))
    args = ap.parse_args(argv)

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    selected = [
        a
        for a in ARTIFACTS
        if any(m in a.local_path for m in args.match) and not any(x in a.local_path for x in args.exclude)
    ]
    if not selected:
        raise SystemExit("no artifacts matched")
    for a in selected:
        bucket, key = _parse_s3_uri(a.remote_uri)
        dst = pathlib.Path(args.destination) / a.local_path
        _download_file(client, bucket, key, dst)  # skips files whose checksum already matches
        print(f"ok  {dst}")


if __name__ == "__main__":
    main()
