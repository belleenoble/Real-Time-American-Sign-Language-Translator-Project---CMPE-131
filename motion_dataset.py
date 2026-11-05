"""Write motion CSVs with source manifests."""

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import uuid


SCHEMA = "legacy_wrist_relative_max_norm_20x63_v1"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    if path.is_symlink():
        raise ValueError(
            f"Symbolic-link outputs are not supported: {path}"
        )

    return path.read_bytes() if path.exists() else None


def _atomic_write(path, data, new=False):
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        if new:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)

    finally:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass


class MotionDataset:
    def __init__(
        self,
        output,
        header,
        append=False,
        signature=None,
    ):
        self.output = Path(output).expanduser().absolute()
        self.manifest = self.output.with_name(
            self.output.name + ".manifest.json"
        )
        self.header = list(header)

        if len(self.header) != 1261 or self.header[-1] != "label":
            raise ValueError(
                "Expected 1260 feature columns followed by label."
            )

        self.signature = dict(signature or {})
        self.original_csv = _read(self.output)
        self.original_manifest = _read(self.manifest)

        self.entries = []
        self.rows = []
        self.known = {}

        if self.original_csv is None:
            if self.original_manifest is not None:
                raise ValueError(
                    "An orphaned manifest exists. Choose a new output."
                )
            return

        if not append:
            raise ValueError(
                "Output already exists. Choose a new --output "
                "or use --append intentionally."
            )

        if self.original_manifest is None:
            raise ValueError(
                "Legacy CSV has no source manifest. "
                "Use a NEW output; do not append to the team's CSV."
            )

        metadata = json.loads(self.original_manifest)

        if not isinstance(metadata, dict):
            raise ValueError("Invalid manifest structure.")

        if (
            metadata.get("version") != 1
            or metadata.get("feature_schema") != SCHEMA
        ):
            raise ValueError(
                "Manifest schema does not match this converter."
            )

        if metadata.get("signature") != self.signature:
            raise ValueError(
                "Conversion settings changed. Use a new output."
            )

        if metadata.get("csv_sha256") != _digest(self.original_csv):
            raise ValueError(
                "CSV/manifest mismatch. Review changes or "
                "restore a matching backup pair."
            )

        reader = csv.reader(
            io.StringIO(self.original_csv.decode("utf-8-sig"))
        )

        if next(reader, None) != self.header:
            raise ValueError("Existing CSV header does not match.")

        entries = metadata.get("entries")

        if not isinstance(entries, list):
            raise ValueError("Manifest entries are missing.")

        count = 0

        for count, row in enumerate(reader, start=1):
            if len(row) != 1261 or row[-1] not in {"J", "Z"}:
                raise ValueError("Invalid row in the existing CSV.")

            if not all(
                math.isfinite(float(value))
                for value in row[:-1]
            ):
                raise ValueError(
                    "Existing CSV contains non-finite features."
                )

            if count > len(entries):
                raise ValueError("Manifest has missing rows.")

            entry = entries[count - 1]
            self._validate_source(entry)

            if (
                entry.get("csv_row") != count + 1
                or entry["label"] != row[-1]
            ):
                raise ValueError(
                    "Manifest source/row mapping does not match."
                )

            checksum = entry["source_sha256"]

            if checksum in self.known:
                raise ValueError(
                    "Manifest contains duplicate source IDs."
                )

            self.known[checksum] = entry

        if (
            count != len(entries)
            or metadata.get("row_count") != count
        ):
            raise ValueError("Manifest row count does not match.")

        self.entries = entries

    @staticmethod
    def _validate_source(record):
        if (
            not isinstance(record, dict)
            or record.get("label") not in {"J", "Z"}
        ):
            raise ValueError("Source label must be J or Z.")

        checksum = record.get("source_sha256")

        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError("Invalid source SHA-256.")

        if any(
            character not in "0123456789abcdef"
            for character in checksum
        ):
            raise ValueError("Invalid source SHA-256.")

        if not isinstance(record.get("source_path"), str):
            raise ValueError("Source path is missing.")

        if (
            not isinstance(record.get("source_bytes"), int)
            or record["source_bytes"] < 0
        ):
            raise ValueError("Invalid source size.")

    def contains(self, record):
        self._validate_source(record)
        previous = self.known.get(record["source_sha256"])

        if (
            previous is not None
            and previous["label"] != record["label"]
        ):
            raise ValueError(
                "Identical video bytes have conflicting J/Z labels."
            )

        return previous is not None

    def add(self, features, record):
        if self.contains(record):
            return False

        if (
            len(features) != 1260
            or not all(
                math.isfinite(float(value))
                for value in features
            )
        ):
            raise ValueError("Expected 1260 finite features.")

        entry = dict(
            record,
            csv_row=len(self.entries) + 2,
        )

        self.rows.append([*features, record["label"]])
        self.entries.append(entry)
        self.known[record["source_sha256"]] = entry

        return True

    def save(self):
        if not self.rows:
            return 0

        self.output.parent.mkdir(parents=True, exist_ok=True)

        lock = self.output.with_name(
            self.output.name + ".lock"
        )
        token = f"{os.getpid()}:{uuid.uuid4().hex}"

        try:
            with lock.open("x", encoding="utf-8") as handle:
                handle.write(token)
        except FileExistsError as error:
            raise ValueError(
                "A dataset lock exists. Check for an active "
                "writer before removing it."
            ) from error

        try:
            if (
                _read(self.output) != self.original_csv
                or _read(self.manifest) != self.original_manifest
            ):
                raise ValueError(
                    "Dataset changed during processing. "
                    "No rows were saved."
                )

            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer)

            if self.original_csv is None:
                writer.writerow(self.header)

            writer.writerows(self.rows)

            prefix = self.original_csv or b""

            if prefix and not prefix.endswith((b"\n", b"\r")):
                prefix += b"\n"

            csv_bytes = prefix + buffer.getvalue().encode("utf-8")

            metadata = {
                "version": 1,
                "feature_schema": SCHEMA,
                "signature": self.signature,
                "csv_sha256": _digest(csv_bytes),
                "row_count": len(self.entries),
                "entries": self.entries,
            }

            manifest_bytes = (
                json.dumps(metadata, indent=2) + "\n"
            ).encode("utf-8")

            backups = []

            if self.original_csv is not None:
                suffix = ".backup-" + uuid.uuid4().hex

                for path, data in (
                    (self.output, self.original_csv),
                    (self.manifest, self.original_manifest),
                ):
                    backup = path.with_name(path.name + suffix)

                    with backup.open("xb") as handle:
                        handle.write(data)

                    backups.append(backup)

            installed = False

            try:
                _atomic_write(
                    self.output,
                    csv_bytes,
                    new=self.original_csv is None,
                )
                installed = True

                _atomic_write(
                    self.manifest,
                    manifest_bytes,
                    new=self.original_manifest is None,
                )

            except OSError:
                if installed and _read(self.output) == csv_bytes:
                    if self.original_csv is None:
                        self.output.unlink()
                    else:
                        os.replace(backups[0], self.output)
                raise

            added = len(self.rows)
            self.original_csv = csv_bytes
            self.original_manifest = manifest_bytes
            self.rows = []

            return added

        finally:
            try:
                if lock.read_text(encoding="utf-8") == token:
                    lock.unlink()
            except OSError:
                pass
            