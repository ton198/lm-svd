"""Download pinned task instructions and C4 documents for task/input sampling."""

import argparse
import codecs
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download


SNI_REPO_ID = "allenai/natural-instructions"
SNI_REVISION = "55a365637381ce7f3748fa2eac7aef1a113bbb82"
SNI_TASK_COUNT = 1613
FLAN_REPO_ID = "google-research/FLAN"
FLAN_REVISION = "e9e4ec6e2701182c7a91af176f705310da541277"
TASKSOURCE_REPO_ID = "tasksource/tasksource-instruct"
TASKSOURCE_REVISION = "4cdba593d93bf1e32c0fd1731d84c6780136fe51"
C4_REPO_ID = "allenai/c4"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
TASKSOURCE_FILES = ("README.md", "data/test-00000-of-00001.parquet")
C4_FILES = ("en/c4-validation.00000-of-00008.json.gz",)
HEADER_FIELDS = (
    "Categories", "Definition", "Input_language", "Output_language",
    "Instruction_language", "Domains",
)
INITIAL_RANGE_BYTES = 65536
MAX_RANGE_BYTES = 1024 * 1024


def fetch_github(url: str, range_bytes: int | None = None, *, tree: bool = False) -> bytes:
    headers = {"User-Agent": "lm-svd-task-source-downloader"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if range_bytes is not None:
        headers["Range"] = f"bytes=0-{range_bytes - 1}"
    attempts = 0
    rate_limit_retried = False
    while True:
        try:
            with urlopen(Request(url, headers=headers), timeout=20) as response:
                # Some raw GitHub responses ignore Range and return the entire file.
                return response.read() if range_bytes is None else response.read(range_bytes)
        except HTTPError as exc:
            if tree and exc.code == 403:
                exc.close()
                if rate_limit_retried:
                    raise RuntimeError("GitHub tree API is rate-limited; set GITHUB_TOKEN") from exc
                rate_limit_retried = True
                time.sleep(60)
                continue
            exc.close()
            attempts += 1
            if attempts >= 3:
                raise
        except (URLError, TimeoutError, OSError):
            attempts += 1
            if attempts >= 3:
                raise
        time.sleep(attempts)


def fetch_sni_header(name: str) -> dict:
    url = (
        f"https://raw.githubusercontent.com/{SNI_REPO_ID}/{SNI_REVISION}/tasks/{name}.json"
    )
    range_bytes = INITIAL_RANGE_BYTES
    while True:
        data = fetch_github(url, range_bytes)
        # A byte range can end inside a UTF-8 character. Keep only complete characters.
        text = codecs.getincrementaldecoder("utf-8")().decode(data, final=False)
        marker = text.find('"Positive Examples"')
        try:
            if marker == -1:
                raise ValueError("missing Positive Examples")
            header_end = text.rfind(",", 0, marker)
            if header_end == -1:
                raise ValueError("missing task header separator")
            header = json.loads(text[:header_end] + "}")
            example_start = text.find("{", marker)
            if example_start == -1:
                raise ValueError("missing first positive example")
            example, _ = json.JSONDecoder().raw_decode(text[example_start:])
            return {
                "name": name,
                **{field: header[field] for field in HEADER_FIELDS},
                "positive_example_input": example["input"],
            }
        except (ValueError, KeyError, TypeError) as exc:
            # Both the header and the first positive example must fit in the range.
            if range_bytes >= MAX_RANGE_BYTES:
                raise RuntimeError(
                    f"cannot parse SNI header and first positive example for {name} "
                    f"within {MAX_RANGE_BYTES} bytes"
                ) from exc
            range_bytes = min(range_bytes * 4, MAX_RANGE_BYTES)


def download_sni(output: Path) -> None:
    if output.exists():
        print(f"Ready: {output}", flush=True)
        return
    tree_url = (
        f"https://api.github.com/repos/{SNI_REPO_ID}/git/trees/{SNI_REVISION}?recursive=1"
    )
    tree = json.loads(fetch_github(tree_url, tree=True))
    paths = sorted(
        entry["path"] for entry in tree["tree"]
        if entry["path"].startswith("tasks/task") and entry["path"].endswith(".json")
    )
    if tree.get("truncated") or len(paths) != SNI_TASK_COUNT:
        raise RuntimeError(
            f"incomplete SNI task tree: truncated={tree.get('truncated')}, "
            f"tasks={len(paths)} (expected {SNI_TASK_COUNT})"
        )
    names = [Path(path).stem for path in paths]
    with ThreadPoolExecutor(max_workers=16) as executor:
        headers = list(executor.map(fetch_sni_header, names))
    headers.sort(key=lambda header: header["name"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for header in headers:
                stream.write(json.dumps(header, ensure_ascii=False) + "\n")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Ready: {output} ({len(headers)} tasks)", flush=True)


def download_flan(output: Path) -> None:
    if not output.exists():
        url = (
            f"https://raw.githubusercontent.com/{FLAN_REPO_ID}/{FLAN_REVISION}/flan/v2/templates.py"
        )
        data = fetch_github(url)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        try:
            temporary.write_bytes(data)
            temporary.replace(output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    print(f"Ready: {output}", flush=True)


def download_dataset(repo_id: str, revision: str, output: Path, files: tuple[str, ...]) -> None:
    if not all((output / name).exists() for name in files):
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=output,
            allow_patterns=list(files),
        )
    for name in files:
        path = output / name
        if not path.exists():
            raise FileNotFoundError(f"download did not produce {path}")
        print(f"Ready: {path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned task instructions and C4 source documents, not generated samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "original",
        help="Root directory for downloaded source files",
    )
    args = parser.parse_args(argv)
    download_sni(args.output_dir / "natural_instructions" / "task_headers.jsonl")
    download_flan(args.output_dir / "flan" / "templates.py")
    download_dataset(
        TASKSOURCE_REPO_ID, TASKSOURCE_REVISION,
        args.output_dir / "tasksource-instruct", TASKSOURCE_FILES,
    )
    download_dataset(C4_REPO_ID, C4_REVISION, args.output_dir / "c4", C4_FILES)
    print("Next: uv run data-pipline build-task-inputs --help", flush=True)


if __name__ == "__main__":
    main()
