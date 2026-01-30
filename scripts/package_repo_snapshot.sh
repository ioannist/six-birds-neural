#!/usr/bin/env bash
set -euo pipefail

# Create a zip of all files not ignored by .gitignore (no git commands).
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$repo_root" python3 - <<'PY'
from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import os
import pathlib
import zipfile

root = pathlib.Path(os.environ["REPO_ROOT"]).resolve()
repo_name = root.name
version_file = root / ".package-repo-snapshot-version"

current_version = -1
if version_file.exists():
    raw = version_file.read_text(encoding="utf-8").strip()
    if raw.isdigit():
        current_version = int(raw)

next_version = current_version + 1
zip_path = root / f"{repo_name}_snapshot_v{next_version}.zip"

@dataclass(frozen=True)
class IgnoreRule:
    base: pathlib.Path
    pattern: str
    negated: bool
    anchored: bool
    dir_only: bool


def _parse_gitignore(path: pathlib.Path) -> list[IgnoreRule]:
    rules: list[IgnoreRule] = []
    if not path.exists():
        return rules
    base = path.parent
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(r"\#") or line.startswith(r"\!"):
            line = line[1:]
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        anchored = line.startswith("/")
        if anchored:
            line = line[1:]
        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1]
        if not line:
            continue
        rules.append(
            IgnoreRule(
                base=base,
                pattern=line,
                negated=negated,
                anchored=anchored,
                dir_only=dir_only,
            )
        )
    return rules


def _match_rule(rule: IgnoreRule, rel_posix: str, is_dir: bool) -> bool:
    if rule.dir_only and not is_dir:
        # Directory rules apply to directories and their contents.
        parts = rel_posix.split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            if _match_rule(
                IgnoreRule(
                    base=rule.base,
                    pattern=rule.pattern,
                    negated=rule.negated,
                    anchored=rule.anchored,
                    dir_only=False,
                ),
                prefix,
                is_dir=True,
            ):
                return True
        return False
    if rule.anchored:
        return fnmatch.fnmatchcase(rel_posix, rule.pattern)
    if "/" in rule.pattern:
        return fnmatch.fnmatchcase(rel_posix, rule.pattern)
    return fnmatch.fnmatchcase(pathlib.PurePosixPath(rel_posix).name, rule.pattern)


def _collect_rules(root_path: pathlib.Path) -> list[IgnoreRule]:
    rules: list[IgnoreRule] = []
    rules.append(
        IgnoreRule(
            base=root_path,
            pattern=".git",
            negated=False,
            anchored=False,
            dir_only=True,
        )
    )
    rules.append(
        IgnoreRule(
            base=root_path,
            pattern="*_snapshot_v*.zip",
            negated=False,
            anchored=False,
            dir_only=False,
        )
    )
    for dirpath, dirnames, filenames in os.walk(root_path):
        if ".git" in dirnames:
            dirnames.remove(".git")
        ignore_path = pathlib.Path(dirpath) / ".gitignore"
        if ignore_path.exists():
            rules.extend(_parse_gitignore(ignore_path))
    return rules


def _is_ignored(path: pathlib.Path, rules: list[IgnoreRule]) -> bool:
    rel_posix = path.relative_to(root).as_posix()
    ignored = False
    for rule in rules:
        try:
            rel_to_rule = path.relative_to(rule.base).as_posix()
        except ValueError:
            continue
        if _match_rule(rule, rel_to_rule, path.is_dir()):
            ignored = not rule.negated
    return ignored


rules = _collect_rules(root)
files: list[pathlib.Path] = []
for path in root.rglob("*"):
    if path.is_dir():
        continue
    if not path.exists():
        continue
    if _is_ignored(path, rules):
        continue
    files.append(path)

if not files:
    raise SystemExit("No files to package (all files ignored).")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in files:
        rel = path.relative_to(root).as_posix()
        zf.write(path, rel)

version_file.write_text(str(next_version), encoding="utf-8")
previous_path = None
if current_version >= 0:
    previous_path = root / f"{repo_name}_snapshot_v{current_version}.zip"
if previous_path and previous_path.exists():
    previous_path.unlink()

print(f"Wrote {zip_path}")
PY
