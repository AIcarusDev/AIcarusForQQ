"""Save a model-visible or public remote image into Agent Linux."""

from __future__ import annotations

import logging
from typing import Any, Union

from llm.media.image_importer import ImageImportError, ImageImporter, MAX_IMAGE_URL_CHARS
from pydantic import Field, RootModel

from tools.contract import ToolArgsModel, ToolContract
from workspace.tools._common import run_on_main_loop


logger = logging.getLogger("AICQ.tools")


class SaveImageRefArgs(ToolArgsModel):
    image_ref: str = Field(min_length=1)
    path: str = Field(min_length=1, pattern=r"^/home/agent/.+")


class SaveImageUrlArgs(ToolArgsModel):
    url: str = Field(min_length=1, max_length=MAX_IMAGE_URL_CHARS, pattern=r"^https?://")
    path: str = Field(min_length=1, pattern=r"^/home/agent/.+")


class SaveImageArgs(RootModel[Union[SaveImageRefArgs, SaveImageUrlArgs]]):
    pass


TOOL_CONTRACT = ToolContract(
    name="save_image",
    description="将 image_ref 或公开 HTTP(S) URL 的图片保存到 linux 目录 /home/agent 下；不覆盖已有文件。",
    args_model=SaveImageArgs,
)

REQUIRES_CONTEXT: list[str] = ["session"]


def _error(exc: ImageImportError) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": exc.code,
        "message": exc.message,
        "retryable": exc.retryable,
    }
    if exc.details:
        error["details"] = exc.details
    return {"ok": False, "error": error}


def make_handler(session: Any):
    async def save(
        workspace_service: Any,
        *,
        path: str,
        image_ref: str | None,
        url: str | None,
    ) -> dict[str, Any]:
        try:
            saved = await ImageImporter(session, workspace_service).save(
                path=path,
                image_ref=image_ref,
                url=url,
            )
        except ImageImportError as exc:
            logger.info("[tools] save_image: 保存失败 code=%s path=%s", exc.code, path)
            return _error(exc)
        except Exception:
            logger.exception("[tools] save_image: 保存异常 path=%s", path)
            return _error(
                ImageImportError(
                    "internal_error",
                    "图片保存失败",
                    retryable=True,
                )
            )
        logger.info(
            "[tools] save_image: 保存完成 path=%s mime=%s size=%d",
            saved.path,
            saved.mime_type,
            saved.size_bytes,
        )
        return {
            "ok": True,
            "path": saved.path,
            "mime_type": saved.mime_type,
            "size_bytes": saved.size_bytes,
        }

    def handler(
        image_ref: str | None = None,
        url: str | None = None,
        path: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        if (image_ref is None) == (url is None):
            return _error(
                ImageImportError(
                    "invalid_arguments",
                    "image_ref 与 url 必须且只能提供一个",
                )
            )

        import app_state

        workspace_service = getattr(app_state, "workspace_service", None)
        if workspace_service is None:
            return _error(
                ImageImportError(
                    "runtime_unavailable",
                    "Agent 电脑服务不可用",
                    retryable=True,
                )
            )
        main_loop = getattr(app_state, "main_loop", None)
        if main_loop is None or not main_loop.is_running():
            return _error(
                ImageImportError(
                    "runtime_unavailable",
                    "主事件循环不可用",
                    retryable=True,
                )
            )
        return run_on_main_loop(
            save(
                workspace_service,
                path=str(path or ""),
                image_ref=image_ref,
                url=url,
            ),
            main_loop,
        )

    return handler
