import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def pin_source(source_train_dir: Path, pinned_dir: Path) -> dict:
    metadata_path = pinned_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        print(f"already pinned: {pinned_dir} (steps {metadata['steps']})")
        return metadata

    filenames = ("metadata.json", "features.pt", "probe.npz")
    for name in filenames:
        path = source_train_dir / name
        if not path.is_file():
            raise ValueError(f"training input does not exist: {path}")

    source = pinned_dir.parent.name
    tmp = pinned_dir.with_name("train.tmp")
    pinned_dir.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        feature_path = source_train_dir / "features.pt"
        source_metadata = source_train_dir / "metadata.json"
        before = (feature_path.stat().st_mtime_ns, source_metadata.stat().st_mtime_ns)
        for name in filenames:
            shutil.copy2(source_train_dir / name, tmp / name)
        after = (feature_path.stat().st_mtime_ns, source_metadata.stat().st_mtime_ns)
        if before == after and before[0] <= before[1]:
            break
    else:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"source {source} changed during pinning 5 times; retry later")

    metadata = json.loads((tmp / "metadata.json").read_text())
    digest = hashlib.sha256()
    with (tmp / "features.pt").open("rb") as features_file:
        for chunk in iter(lambda: features_file.read(1024 * 1024), b""):
            digest.update(chunk)
    pin = {
        "source": source,
        "source_train_dir": str(source_train_dir),
        "steps": metadata["steps"],
        "pinned_at": datetime.now(timezone.utc).isoformat(),
        "features_sha256": digest.hexdigest(),
    }
    (tmp / "pin.json").write_text(json.dumps(pin, indent=2) + "\n")
    os.replace(tmp, pinned_dir)
    return metadata
