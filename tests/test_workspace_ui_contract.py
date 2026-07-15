from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_workspace_picker_stages_and_persists_the_selected_path() -> None:
    source = (ROOT / "src" / "templates" / "settings.html").read_text(encoding="utf-8")

    assert "beginWorkspaceDirectoryWatch(selectionId);" in source
    assert "/api/workspace/directory-selections/" in source
    assert "await persistWorkspaceDirectory(selection.path);" in source
    assert 'await api.choose_workspace_directory(value("workspace_install_root"), selectionId)' in source
    assert "await saveWorkspaceConfig({ silent: true })" in source


def test_launcher_workspace_picker_has_return_value_and_webview_callback_paths() -> None:
    source = (ROOT / "launcher.py").read_text(encoding="utf-8")

    assert 'selection_id: str = ""' in source
    assert 'getattr(file_dialog, "FOLDER", None)' in source
    assert "publish_workspace_directory_selection" in source
    assert '_publish_selection(status="selected", path=selected_path)' in source
    assert 'return {"ok": True, "path": selected_path}' in source


def test_partial_workspace_install_is_rendered_and_retried_as_a_build() -> None:
    source = (ROOT / "src" / "templates" / "settings.html").read_text(encoding="utf-8")

    assert 'observed.partial_install) primary.textContent = "修复并继续构建"' in source
    assert 'saved.observed?.partial_install || stateName === "not_built" ? "build" : "apply"' in source
    assert "可安全恢复的安装半成品" in source


def test_workspace_refresh_loads_a_new_terminal_job_log_once() -> None:
    source = (ROOT / "src" / "templates" / "settings.html").read_text(encoding="utf-8")

    assert "activeJob || workspaceUi.jobId !== payload.job.job_id" in source
    assert "beginWorkspaceJobPolling(payload.job.job_id, false)" in source


def test_workspace_resumable_partial_uses_explicit_continue_label() -> None:
    source = (ROOT / "src" / "templates" / "settings.html").read_text(encoding="utf-8")

    assert 'observed.partial_repair_mode === "resume"' in source
    assert 'primary.textContent = "从失败阶段继续构建"' in source
