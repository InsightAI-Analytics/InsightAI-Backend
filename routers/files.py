"""
Files router — handles file upload and listing.
"""

from __future__ import annotations
import logging
from typing import Annotated

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from models.schemas import ColumnInfo, DatasetInfo, UploadResponse
from services.parser import ParseError, parse_file, get_file_metadata
from services.normalizer import normalize, get_schema_description
import session_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/files", tags=["files"])

MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024  # bytes


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    x_session_id: Annotated[str | None, Header()] = None,
) -> UploadResponse:
    """
    Upload one or more CSV/XLSX files.
    Returns session_id, info about each uploaded file, and any parse errors.
    """
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required.")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    existing = session_store.get_datasets(x_session_id)
    if len(existing) + len(files) > settings.max_files_per_session:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_files_per_session} files per session allowed.",
        )

    uploaded: list[DatasetInfo] = []
    errors: list[dict[str, str]] = []

    for upload in files:
        filename = upload.filename or "unnamed"
        try:
            content = await upload.read()

            # Size check
            if len(content) > MAX_FILE_SIZE:
                errors.append(
                    {
                        "file": filename,
                        "error": f"File exceeds size limit of {settings.max_file_size_mb} MB.",
                    }
                )
                continue

            # Duplicate check
            existing_names = {d["name"] for d in session_store.get_datasets(x_session_id)}
            if filename in existing_names:
                errors.append(
                    {"file": filename, "error": "A file with this name is already uploaded."}
                )
                continue

            # Parse
            df = parse_file(filename, content)

            # Normalize
            df_norm, original_columns = normalize(df)

            # Metadata
            meta = get_file_metadata(df_norm, original_columns)

            # Store in session
            session_store.add_dataset(
                x_session_id,
                {
                    "name": filename,
                    "df": df_norm,
                    "original_columns": original_columns,
                    "metadata": meta,
                },
            )

            column_info = [
                ColumnInfo(
                    name=c["name"],
                    original_name=c["original_name"],
                    dtype=c["dtype"],
                )
                for c in meta["column_info"]
            ]

            uploaded.append(
                DatasetInfo(
                    name=filename,
                    rows=meta["rows"],
                    columns=meta["columns"],
                    column_info=column_info,
                    sample=meta["sample"],
                )
            )
            logger.info("Uploaded '%s' (%d rows) for session %s", filename, meta["rows"], x_session_id)

        except ParseError as exc:
            logger.warning("Parse error for '%s': %s", filename, exc)
            errors.append({"file": filename, "error": str(exc)})
        except Exception as exc:
            logger.exception("Unexpected error processing '%s'", filename)
            errors.append({"file": filename, "error": f"Unexpected error: {exc}"})

    return UploadResponse(
        session_id=x_session_id,
        uploaded=uploaded,
        errors=errors,
    )


@router.get("", response_model=list[DatasetInfo])
async def list_files(
    x_session_id: Annotated[str | None, Header()] = None,
) -> list[DatasetInfo]:
    """List all uploaded files for the current session."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required.")

    datasets = session_store.get_datasets(x_session_id)
    result = []
    for d in datasets:
        meta = d["metadata"]
        column_info = [
            ColumnInfo(
                name=c["name"],
                original_name=c["original_name"],
                dtype=c["dtype"],
            )
            for c in meta["column_info"]
        ]
        result.append(
            DatasetInfo(
                name=d["name"],
                rows=meta["rows"],
                columns=meta["columns"],
                column_info=column_info,
                sample=meta["sample"],
            )
        )
    return result


@router.delete("/{filename}")
async def delete_file(
    filename: str,
    x_session_id: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """Remove an uploaded file from the session."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required.")

    removed = session_store.remove_dataset(x_session_id, filename)
    if not removed:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return JSONResponse({"message": f"File '{filename}' removed successfully."})
