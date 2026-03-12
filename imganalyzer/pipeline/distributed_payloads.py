"""Helpers for DB-less distributed worker job packaging and result persistence."""
from __future__ import annotations

import base64
import sqlite3
from typing import Any

from imganalyzer.db.repository import Repository
from imganalyzer.pipeline.modules import write_local_ai_split_tables

# Fields managed by the coordinator DB layer; worker payloads must not persist
# these values directly when replaying results into a different database context.
_ANALYSIS_HOUSEKEEPING_KEYS = {"id", "image_id", "analyzed_at", "provider"}


def _clean_analysis_row(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict) or not data:
        return {}
    return {
        key: value for key, value in data.items() if key not in _ANALYSIS_HOUSEKEEPING_KEYS
    }


def _encode_blob(value: bytes | None) -> str | None:
    if value is None:
        return None
    return base64.b64encode(value).decode("ascii")


def _decode_blob(value: Any) -> bytes | None:
    if value in (None, ""):
        return None
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise TypeError("Expected base64 string for binary payload")
    return base64.b64decode(value.encode("ascii"))


def seed_job_context(
    conn: sqlite3.Connection,
    repo: Repository,
    *,
    image_id: int,
    file_path: str,
    image_info: dict[str, Any] | None,
    context: dict[str, Any] | None,
) -> None:
    """Populate a temporary SQLite job sandbox with the claimed image context."""
    conn.execute(
        "INSERT INTO images (id, file_path) VALUES (?, ?)",
        [image_id, file_path],
    )
    if image_info:
        repo.update_image(
            image_id,
            width=image_info.get("width"),
            height=image_info.get("height"),
            format=image_info.get("format"),
        )

    modules = context.get("modules", {}) if isinstance(context, dict) else {}

    local_ai = modules.get("local_ai")
    if isinstance(local_ai, dict) and local_ai:
        repo.upsert_local_ai(image_id, _clean_analysis_row(local_ai))

    objects = modules.get("objects")
    if isinstance(objects, dict) and objects:
        repo.upsert_objects(image_id, _clean_analysis_row(objects))

    aesthetic = modules.get("aesthetic")
    if isinstance(aesthetic, dict) and aesthetic:
        repo.upsert_aesthetic(image_id, _clean_analysis_row(aesthetic))

    perception = modules.get("perception")
    if isinstance(perception, dict) and perception:
        repo.upsert_perception(image_id, _clean_analysis_row(perception))

    cloud_ai = modules.get("cloud_ai")
    if isinstance(cloud_ai, dict):
        providers = cloud_ai.get("providers", [])
        if isinstance(providers, list):
            for provider_row in providers:
                if not isinstance(provider_row, dict):
                    continue
                provider = str(provider_row.get("provider", "")).strip()
                if not provider:
                    continue
                repo.upsert_cloud_ai(image_id, provider, _clean_analysis_row(provider_row))

    conn.commit()


def extract_result_payload(
    conn: sqlite3.Connection,
    repo: Repository,
    *,
    image_id: int,
    module: str,
) -> dict[str, Any]:
    """Serialize the module result produced in a temporary worker sandbox."""
    image = repo.get_image(image_id) or {}
    payload: dict[str, Any] = {
        "image": {
            "width": image.get("width"),
            "height": image.get("height"),
            "format": image.get("format"),
        }
    }

    if module == "embedding":
        rows = conn.execute(
            """SELECT embedding_type, vector, model_version
               FROM embeddings
               WHERE image_id = ?""",
            [image_id],
        ).fetchall()
        payload["embeddings"] = [
            {
                "embeddingType": row["embedding_type"],
                "vector": _encode_blob(row["vector"]),
                "modelVersion": row["model_version"],
            }
            for row in rows
            if row["vector"] is not None
        ]
        return payload

    if module == "cloud_ai":
        payload["data"] = repo.get_analysis(image_id, "cloud_ai") or {"providers": []}
        aesthetic = repo.get_analysis(image_id, "aesthetic")
        if aesthetic:
            payload["aesthetic"] = _clean_analysis_row(aesthetic)
        blip2 = repo.get_analysis(image_id, "blip2")
        if blip2:
            payload["blip2"] = _clean_analysis_row(blip2)
        return payload

    if module == "aesthetic":
        payload["data"] = _clean_analysis_row(repo.get_analysis(image_id, "aesthetic"))
        perception = repo.get_analysis(image_id, "perception")
        if perception:
            payload["perception"] = _clean_analysis_row(perception)
        return payload

    data = repo.get_analysis(image_id, module)
    payload["data"] = _clean_analysis_row(data)

    if module == "faces":
        rows = conn.execute(
            """SELECT face_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                      embedding, age, gender, identity_name, det_score, thumbnail
               FROM face_occurrences
               WHERE image_id = ?
               ORDER BY face_idx""",
            [image_id],
        ).fetchall()
        payload["faceOccurrences"] = [
            {
                "face_idx": row["face_idx"],
                "bbox_x1": row["bbox_x1"],
                "bbox_y1": row["bbox_y1"],
                "bbox_x2": row["bbox_x2"],
                "bbox_y2": row["bbox_y2"],
                "embedding": _encode_blob(row["embedding"]),
                "age": row["age"],
                "gender": row["gender"],
                "identity_name": row["identity_name"],
                "det_score": row["det_score"],
                "thumbnail": _encode_blob(row["thumbnail"]),
            }
            for row in rows
        ]

    return payload


def persist_result_payload(
    conn: sqlite3.Connection,
    repo: Repository,
    *,
    image_id: int,
    module: str,
    payload: dict[str, Any],
) -> None:
    """Persist a worker-reported module payload into the coordinator database."""
    image_info = payload.get("image")
    if isinstance(image_info, dict):
        repo.update_image(
            image_id,
            width=image_info.get("width"),
            height=image_info.get("height"),
            format=image_info.get("format"),
        )

    if module == "metadata":
        data = _clean_analysis_row(payload.get("data"))
        if data:
            repo.upsert_metadata(image_id, data)
        return

    if module == "technical":
        data = _clean_analysis_row(payload.get("data"))
        if data:
            repo.upsert_technical(image_id, data)
        return

    if module == "local_ai":
        data = _clean_analysis_row(payload.get("data"))
        if not data:
            return
        repo.upsert_local_ai(image_id, data)
        write_local_ai_split_tables(conn, repo, image_id, data, wrap_transactions=False)
        return

    if module == "objects":
        data = _clean_analysis_row(payload.get("data"))
        if data:
            repo.upsert_objects(image_id, data)
        return

    if module == "faces":
        data = _clean_analysis_row(payload.get("data"))
        if not data:
            return
        repo.upsert_faces(image_id, data)
        face_occurrences = payload.get("faceOccurrences", [])
        if isinstance(face_occurrences, list):
            decoded: list[dict[str, Any]] = []
            for occurrence in face_occurrences:
                if not isinstance(occurrence, dict):
                    continue
                decoded.append(
                    {
                        "face_idx": occurrence.get("face_idx"),
                        "bbox_x1": occurrence.get("bbox_x1"),
                        "bbox_y1": occurrence.get("bbox_y1"),
                        "bbox_x2": occurrence.get("bbox_x2"),
                        "bbox_y2": occurrence.get("bbox_y2"),
                        "embedding": _decode_blob(occurrence.get("embedding")),
                        "age": occurrence.get("age"),
                        "gender": occurrence.get("gender"),
                        "identity_name": occurrence.get("identity_name", "Unknown"),
                        "det_score": occurrence.get("det_score"),
                        "thumbnail": _decode_blob(occurrence.get("thumbnail")),
                    }
                )
            repo.upsert_face_occurrences(image_id, decoded)
        return

    if module == "cloud_ai":
        cloud_data = payload.get("data", {})
        if isinstance(cloud_data, dict):
            providers = cloud_data.get("providers", [])
            if isinstance(providers, list):
                for provider_row in providers:
                    if not isinstance(provider_row, dict):
                        continue
                    provider = str(provider_row.get("provider", "")).strip()
                    if not provider:
                        continue
                    repo.upsert_cloud_ai(image_id, provider, _clean_analysis_row(provider_row))
        aesthetic = payload.get("aesthetic")
        if isinstance(aesthetic, dict) and aesthetic:
            repo.upsert_aesthetic(image_id, _clean_analysis_row(aesthetic))
        blip2 = payload.get("blip2")
        if isinstance(blip2, dict) and blip2:
            repo.upsert_blip2(image_id, _clean_analysis_row(blip2))
        return

    if module == "aesthetic":
        data = _clean_analysis_row(payload.get("data"))
        if data:
            repo.upsert_aesthetic(image_id, data)
        perception = _clean_analysis_row(payload.get("perception"))
        if perception:
            repo.upsert_perception(image_id, perception)
        return

    if module == "embedding":
        embeddings = payload.get("embeddings", [])
        if not isinstance(embeddings, list):
            raise ValueError("Embedding payload must contain a list of vectors")
        for item in embeddings:
            if not isinstance(item, dict):
                continue
            vector = _decode_blob(item.get("vector"))
            embedding_type = str(item.get("embeddingType", "")).strip()
            if vector is None or not embedding_type:
                continue
            repo.upsert_embedding(
                image_id,
                embedding_type,
                vector,
                str(item.get("modelVersion", "")),
            )
        return

    raise ValueError(f"Unsupported distributed module payload: {module}")
