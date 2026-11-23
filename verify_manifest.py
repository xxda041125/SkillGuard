"""Validate the paper split and optionally verify every document's SHA-256 digest."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="Directory containing the manifest's files")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text(encoding="utf-8"))
    expected = {"train": (160, 2513), "val": (20, 320), "test": (20, 319)}
    split_for = {}
    clean_for = {}
    paths = set()
    counts = Counter()
    for split in expected:
        for identity in manifest["splits"][split]["skill_ids"]:
            require(identity not in split_for, f"Repeated split membership: {identity}")
            split_for[identity] = split
    for role in ("clean", "attacked"):
        for row in manifest[role]:
            identity = row["group"] + "/" + row["skill_name"]
            split = split_for[identity]
            path = row["path"]
            require(path not in paths, f"Duplicate path: {path}")
            paths.add(path)
            if role == "clean":
                require(identity not in clean_for, f"Multiple clean documents: {identity}")
                clean_for[identity] = path
            else:
                require(clean_for.get(identity) == row["clean_path"], f"Unpaired document: {path}")
                require(row["generator_id"] in manifest["splits"][split]["generators"], path)
            counts[split, role] += 1
            if args.corpus:
                actual = hashlib.sha256((args.corpus / path).read_bytes()).hexdigest()
                require(actual == row["sha256"], f"File digest mismatch: {path}")
    require(set(split_for) == set(clean_for), "Split membership differs from the clean index")
    for split, target in expected.items():
        result = (counts[split, "clean"], counts[split, "attacked"])
        require(result == target, f"{split}: expected {target}, found {result}")
        print(f"{split}: {result[0]} identities, {result[1]} pairs")
    if args.corpus:
        print(f"Verified {len(paths)} document digests.")


if __name__ == "__main__":
    main()
