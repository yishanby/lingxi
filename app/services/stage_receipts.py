"""Typed memory-stage results and content-free durable rebuild receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal, Sequence

from app.services.md_store import ChatRecord, render_chat_records

if TYPE_CHECKING:
    from app.services.md_store import MarkdownMemoryTransaction

StageName = Literal["story", "memory", "episode", "summary", "rag", "assets"]
RECEIPT_STAGES = frozenset({"story", "memory", "summary", "assets"})
_RECEIPT_VERSION = 1
RECEIPT_MAX_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class ChatSourceIdentity:
    count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: str
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class StageUpdateResult:
    stage: StageName
    completed: bool
    source: ChatSourceIdentity
    checkpoint: int
    artifacts: tuple[ArtifactIdentity, ...] = ()
    inputs: tuple[ArtifactIdentity, ...] = ()


@dataclass(frozen=True, slots=True)
class StageReceipt:
    version: int
    stage: StageName
    source: ChatSourceIdentity
    checkpoint: int
    artifacts: tuple[ArtifactIdentity, ...]
    inputs: tuple[ArtifactIdentity, ...]


def chat_source_identity(records: Sequence[ChatRecord]) -> ChatSourceIdentity:
    rendered = render_chat_records(list(records)).encode("utf-8")
    return ChatSourceIdentity(
        count=len(records),
        sha256=hashlib.sha256(rendered).hexdigest(),
    )


def text_artifact(path: str, text: str) -> ArtifactIdentity:
    encoded = text.encode("utf-8")
    return ArtifactIdentity(
        path=path,
        sha256=hashlib.sha256(encoded).hexdigest(),
        byte_size=len(encoded),
    )


def receipt_path(stage: StageName) -> str:
    if stage not in RECEIPT_STAGES:
        raise ValueError("stage does not use a rebuild receipt")
    return f"rebuild_receipts/{stage}.json"


def _validate_source(source: ChatSourceIdentity) -> None:
    if type(source.count) is not int or source.count < 0:
        raise ValueError("invalid stage source count")
    if (
        not isinstance(source.sha256, str)
        or len(source.sha256) != 64
        or any(character not in "0123456789abcdef" for character in source.sha256)
    ):
        raise ValueError("invalid stage source hash")


def _validate_artifacts(artifacts: tuple[ArtifactIdentity, ...]) -> None:
    if not isinstance(artifacts, tuple):
        raise ValueError("invalid stage artifacts")
    paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, ArtifactIdentity):
            raise ValueError("invalid stage artifacts")
        if (
            not isinstance(artifact.path, str)
            or not artifact.path
            or "\\" in artifact.path
            or ":" in artifact.path
        ):
            raise ValueError("invalid stage artifact path")
        normalized = PurePosixPath(artifact.path)
        if (
            normalized.is_absolute()
            or normalized.as_posix() != artifact.path
            or ".." in normalized.parts
        ):
            raise ValueError("invalid stage artifact path")
        if artifact.path in paths:
            raise ValueError("invalid stage artifacts")
        paths.add(artifact.path)
        if (
            not isinstance(artifact.sha256, str)
            or len(artifact.sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in artifact.sha256
            )
        ):
            raise ValueError("invalid stage artifact hash")
        if type(artifact.byte_size) is not int or artifact.byte_size < 0:
            raise ValueError("invalid stage artifact byte size")


def validate_stage_result(result: StageUpdateResult, stage: StageName) -> None:
    if not isinstance(result, StageUpdateResult) or result.stage != stage:
        raise RuntimeError(f"{stage} stage did not return a completed result")
    if result.completed is not True:
        raise RuntimeError(f"{stage} stage did not return a completed result")
    _validate_source(result.source)
    if type(result.checkpoint) is not int or result.checkpoint < 0:
        raise RuntimeError(f"{stage} stage did not return a completed result")
    _validate_artifacts(result.artifacts)
    _validate_artifacts(result.inputs)


def render_receipt(receipt: StageReceipt) -> str:
    if (
        type(receipt.version) is not int
        or receipt.version != _RECEIPT_VERSION
        or receipt.stage not in RECEIPT_STAGES
    ):
        raise ValueError("invalid stage receipt")
    _validate_source(receipt.source)
    if type(receipt.checkpoint) is not int or receipt.checkpoint < 0:
        raise ValueError("invalid stage receipt")
    _validate_artifacts(receipt.artifacts)
    _validate_artifacts(receipt.inputs)

    def render_artifact(artifact: ArtifactIdentity) -> dict[str, object]:
        return {
            "path": artifact.path,
            "sha256": artifact.sha256,
            "byte_size": artifact.byte_size,
        }

    document = {
        "version": receipt.version,
        "stage": receipt.stage,
        "source_count": receipt.source.count,
        "source_sha256": receipt.source.sha256,
        "checkpoint": receipt.checkpoint,
        "artifacts": [render_artifact(artifact) for artifact in receipt.artifacts],
        "inputs": [render_artifact(artifact) for artifact in receipt.inputs],
    }
    rendered = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"
    if len(rendered.encode("utf-8")) > RECEIPT_MAX_BYTES:
        raise ValueError("invalid stage receipt")
    return rendered


def parse_receipt(text: str) -> StageReceipt:
    if len(text.encode("utf-8")) > RECEIPT_MAX_BYTES:
        raise ValueError("invalid stage receipt")

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError
            parsed[key] = value
        return parsed

    try:
        document = json.loads(text, object_pairs_hook=strict_object)
        if not isinstance(document, dict) or set(document) != {
            "version",
            "stage",
            "source_count",
            "source_sha256",
            "checkpoint",
            "artifacts",
            "inputs",
        }:
            raise ValueError

        def parse_artifacts(key: str) -> tuple[ArtifactIdentity, ...]:
            raw_artifacts = document[key]
            if not isinstance(raw_artifacts, list):
                raise ValueError
            artifacts = tuple(
                ArtifactIdentity(
                    path=item["path"],
                    sha256=item["sha256"],
                    byte_size=item["byte_size"],
                )
                for item in raw_artifacts
                if isinstance(item, dict)
                and set(item) == {"path", "sha256", "byte_size"}
            )
            if len(artifacts) != len(raw_artifacts):
                raise ValueError
            return artifacts

        receipt = StageReceipt(
            version=document["version"],
            stage=document["stage"],
            source=ChatSourceIdentity(
                count=document["source_count"],
                sha256=document["source_sha256"],
            ),
            checkpoint=document["checkpoint"],
            artifacts=parse_artifacts("artifacts"),
            inputs=parse_artifacts("inputs"),
        )
        render_receipt(receipt)
        return receipt
    except (KeyError, TypeError, ValueError, RecursionError, json.JSONDecodeError):
        raise ValueError("invalid stage receipt") from None


async def write_receipt(
    transaction: MarkdownMemoryTransaction,
    result: StageUpdateResult,
) -> None:
    validate_stage_result(result, result.stage)
    receipt = StageReceipt(
        version=_RECEIPT_VERSION,
        stage=result.stage,
        source=result.source,
        checkpoint=result.checkpoint,
        artifacts=result.artifacts,
        inputs=result.inputs,
    )
    await transaction.write_text(receipt_path(result.stage), render_receipt(receipt))
