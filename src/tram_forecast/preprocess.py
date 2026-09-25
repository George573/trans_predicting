"""Bounded disk preparation. No exploratory full-data profiling or training."""

import shutil
import tempfile
import time
from datetime import date
from pathlib import Path

import duckdb
import numpy as np

from .config import FIELDS
from .io import digest, source_fingerprints, write_json
from .schema import count_scale, label_grid

SCHEMA_VERSION = 1


def preparation_identity(settings, regime):
    if regime not in ("validation", "final"):
        raise ValueError("unknown preparation regime")
    data = settings.data
    end = data.validation_cutoff if regime == "validation" else data.final_cutoff
    return {
        "schema_version": SCHEMA_VERSION,
        "regime": regime,
        "start": data.start,
        "end": end,
        "data": settings.to_dict()["data"],
        "category_caps": settings.model.category_caps,
        "min_frequency": settings.model.min_frequency,
        "fields": FIELDS,
        "sources": source_fingerprints((*data.raw_paths, *data.label_paths)),
    }


def prepare(settings, regime, output=None):
    """Publish immutable artifacts atomically; refuse to overwrite incompatible data."""
    identity = preparation_identity(settings, regime)
    fingerprint = digest(identity)
    output = Path(output or Path(settings.data.output_root) / regime)
    if output.exists():
        from .storage import Store

        existing = Store(output, events=False)
        if existing.metadata["fingerprint"] != fingerprint:
            raise ValueError(
                f"{output}: incompatible existing artifact; use a new output directory"
            )
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(settings.data.temp_dir)
    temp_root.mkdir(parents=True, exist_ok=True)
    for root in (output.parent, temp_root):
        if shutil.disk_usage(root).free < settings.data.min_free_disk_bytes:
            raise OSError(f"insufficient free space in {root}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    started = time.monotonic()
    try:
        _build(settings, identity, fingerprint, stage, temp_root)
        if (
            source_fingerprints((*settings.data.raw_paths, *settings.data.label_paths))
            != identity["sources"]
        ):
            raise ValueError("source files changed during preparation")
        stage.rename(output)
        print(f"Prepared {output} in {time.monotonic() - started:.1f}s", flush=True)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return output


def _build(settings, identity, fingerprint, stage, temp_root):
    data = settings.data
    routes = tuple(data.routes)
    start = date.fromisoformat(identity["start"])
    end = date.fromisoformat(identity["end"])
    # Route 5 is deliberately checked even though it has no neural index.
    counts, presence = label_grid(data.label_paths, (*routes, 5), start, end)
    if presence[-1].any():
        raise ValueError(
            "route 5 has labels: conflicts with the no-history fallback policy"
        )
    counts = counts[:-1]
    presence = presence[:-1]
    if not presence.any(axis=1).all():
        raise ValueError("a configured neural route has no fitting-period labels")
    np.save(stage / "boardings.npy", counts)
    np.save(stage / "label_present.npy", presence)
    scale = count_scale(counts)
    write_json(
        stage / "scaling.json", {"scale": scale, "start": str(start), "end": str(end)}
    )
    eval_end = date.fromisoformat(
        data.final_cutoff if identity["regime"] == "validation" else data.forecast_end
    )
    if identity["regime"] == "validation":
        targets, target_present = label_grid(
            data.label_paths, (*routes, 5), end, eval_end
        )
        np.save(stage / "evaluation.npy", targets)
        np.save(stage / "evaluation_present.npy", target_present)
    with tempfile.TemporaryDirectory(prefix="tram-sort-", dir=temp_root) as work:
        connection = duckdb.connect(str(Path(work) / "staging.duckdb"))
        try:
            connection.execute("SET memory_limit = ?", [data.memory_limit])
            connection.execute("SET threads = ?", [data.threads])
            connection.execute("SET temp_directory = ?", [str(Path(work) / "spill")])
            columns = ", ".join(f"{f} VARCHAR" for f in FIELDS)
            connection.execute(
                f"CREATE TABLE events(route INTEGER, stamp TIMESTAMP, file_no INTEGER, row_no BIGINT, {columns})"
            )
            diagnostics = []
            for file_no, path in enumerate(data.raw_paths):
                print(
                    f"Staging raw file {file_no + 1}/{len(data.raw_paths)}: {path}",
                    flush=True,
                )
                # parallel=false and insertion order preserve a deterministic source-row tie key.
                connection.execute(
                    "CREATE OR REPLACE TEMP TABLE input AS SELECT row_number() OVER () AS source_row, * FROM read_csv(?, delim=';', header=true, all_varchar=true, parallel=false, nullstr='', strict_mode=true)",
                    [str(path)],
                )
                names = {
                    row[0] for row in connection.execute("DESCRIBE input").fetchall()
                }
                if not set(FIELDS + ("ngpt_route", "tran_date_time")).issubset(names):
                    raise ValueError(f"{path}: missing event columns")
                invalid = connection.execute(
                    "SELECT source_row FROM input WHERE ngpt_route IS NULL OR NOT regexp_full_match(trim(ngpt_route), '[0-9]+(\\s+трамвай)?') OR try_cast(regexp_extract(trim(ngpt_route), '^[0-9]+') AS INTEGER) IS NULL OR tran_date_time IS NULL OR try_cast(tran_date_time AS TIMESTAMP) IS NULL OR regexp_matches(tran_date_time, '(Z|[+-][0-9]{2}:?[0-9]{2})$') LIMIT 1"
                ).fetchone()
                if invalid:
                    raise ValueError(
                        f"{path}: invalid route or naive timestamp near source row {invalid[0] + 1}"
                    )
                connection.execute(
                    "CREATE OR REPLACE TEMP VIEW parsed AS SELECT cast(regexp_extract(trim(ngpt_route), '^[0-9]+') AS INTEGER) AS route, cast(tran_date_time AS TIMESTAMP) AS stamp, source_row, "
                    + ", ".join(f"nullif(trim({f}),'') AS {f}" for f in FIELDS)
                    + " FROM input"
                )
                total = connection.execute("SELECT count(*) FROM parsed").fetchone()[0]
                selected = connection.execute(
                    "SELECT count(*) FROM parsed WHERE stamp>=? AND stamp<? AND route IN (SELECT unnest(?))",
                    [start, end, list(routes)],
                ).fetchone()[0]
                connection.execute(
                    "INSERT INTO events SELECT route,stamp,?,source_row,"
                    + ",".join(FIELDS)
                    + " FROM parsed WHERE stamp>=? AND stamp<? AND route IN (SELECT unnest(?))",
                    [file_no, start, end, list(routes)],
                )
                diagnostics.append(
                    {
                        "path": str(path),
                        "rows": total,
                        "selected": selected,
                        "excluded": total - selected,
                    }
                )
                connection.execute("DROP VIEW parsed")
                connection.execute("DROP TABLE input")
            vocab = {}
            category_diagnostics = {}
            for f, cap in zip(FIELDS, settings.model.category_caps):
                kept = connection.execute(
                    f"SELECT {f}, count(*) AS n FROM events WHERE {f} IS NOT NULL GROUP BY {f} HAVING count(*)>=? ORDER BY n DESC, {f} ASC LIMIT ?",
                    [settings.model.min_frequency, cap],
                ).fetchall()
                vocab[f] = {value: i + 3 for i, (value, _) in enumerate(kept)}
                connection.execute(
                    f"CREATE TEMP TABLE vocab_{f}(value VARCHAR, id INTEGER)"
                )
                if kept:
                    connection.executemany(
                        f"INSERT INTO vocab_{f} VALUES (?,?)", list(vocab[f].items())
                    )
                missing, unknown = connection.execute(
                    f"SELECT count(*) FILTER (WHERE e.{f} IS NULL), count(*) FILTER (WHERE e.{f} IS NOT NULL AND v.id IS NULL) FROM events e LEFT JOIN vocab_{f} v ON e.{f}=v.value"
                ).fetchone()
                category_diagnostics[f] = {
                    "retained": len(kept),
                    "missing": missing,
                    "unknown_or_rare": unknown,
                }
            write_json(stage / "vocab.json", vocab)
            n = connection.execute("SELECT count(*) FROM events").fetchone()[0]
            ids = (
                np.lib.format.open_memmap(
                    stage / "events.npy", mode="w+", dtype=np.int32, shape=(n, 5)
                )
                if n
                else np.empty((0, 5), dtype=np.int32)
            )
            width = counts.shape[1]
            offsets = np.zeros((len(routes), width, 2), dtype=np.int64)
            hour_counts = np.zeros((len(routes), width), dtype=np.int64)
            route_index = {r: i for i, r in enumerate(routes)}
            group_rows = connection.execute(
                "SELECT route, date_diff('hour', ?::TIMESTAMP, date_trunc('hour',stamp)), count(*) FROM events GROUP BY 1,2",
                [start],
            ).fetchall()
            for route, hour, count in group_rows:
                hour_counts[route_index[route], hour] = count
            ends = np.cumsum(hour_counts.ravel(), dtype=np.int64)
            offsets[:, :, 1] = ends.reshape(hour_counts.shape)
            offsets[:, :, 0] = (ends - hour_counts.ravel()).reshape(hour_counts.shape)
            np.save(stage / "offsets.npy", offsets)
            expression = ",".join(
                f"CASE WHEN e.{f} IS NULL THEN 1 ELSE coalesce(v{i}.id,2) END"
                for i, f in enumerate(FIELDS)
            )
            joins = " ".join(
                f"LEFT JOIN vocab_{f} v{i} ON e.{f}=v{i}.value"
                for i, f in enumerate(FIELDS)
            )
            # Offset route order follows configured routes, not an accidental numeric sort.
            route_order = (
                "CASE e.route "
                + " ".join(f"WHEN {r} THEN {i}" for i, r in enumerate(routes))
                + " END"
            )
            cursor = connection.execute(
                f"SELECT {expression} FROM events e {joins} ORDER BY {route_order}, e.stamp,e.file_no,e.row_no"
            )
            position = 0
            while batch := cursor.fetchmany(data.fetch_rows):
                ids[position : position + len(batch)] = np.asarray(
                    batch, dtype=np.int32
                )
                position += len(batch)
            if position != n:
                raise RuntimeError("event export row count mismatch")
            if n:
                ids.flush()
            else:
                np.save(stage / "events.npy", ids)
            successful = np.zeros_like(counts, dtype=np.int64)
            for route, hour, count in connection.execute(
                "SELECT route,date_diff('hour',?::TIMESTAMP,date_trunc('hour',stamp)),count(*) FROM events WHERE validation_result='1' GROUP BY 1,2",
                [start],
            ).fetchall():
                successful[route_index[route], hour] = count
            mismatch = int(np.count_nonzero(successful != counts))
            print(
                f"Exported {n} events; {mismatch} route-hours differ from supplied boarding labels",
                flush=True,
            )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "identity": identity,
                "regime": identity["regime"],
                "start": str(start),
                "end": str(end),
                "evaluation_end": str(eval_end),
                "routes": routes,
                "events": n,
                "vocab_sizes": [len(vocab[f]) + 3 for f in FIELDS],
                "source_rows": diagnostics,
                "categories": category_diagnostics,
                "max_events_per_hour": int(hour_counts.max()),
                "event_rows_by_route": hour_counts.sum(axis=1).tolist(),
                "label_rows_by_route": presence.sum(axis=1).tolist(),
                "raw_label_mismatched_hours": mismatch,
                "raw_label_absolute_difference": float(
                    np.abs(successful - counts).sum()
                ),
                "scale": scale,
            }
            write_json(stage / "metadata.json", metadata)
        finally:
            connection.close()
