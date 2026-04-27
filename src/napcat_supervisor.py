# Copyright (C) 2026  AIcarusDev
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""napcat_supervisor.py — NapCat 自动重启 + 二维码捕获

职责：
    1. 接收掉线告警事件，按冷却 + 每小时熔断策略拉起 NapCat 启动批处理。
    2. 重启后等待 recovery_grace_seconds 秒观察心跳是否自动恢复。
    3. 仍未恢复时扫描指定 glob 路径下重启时刻之后产生的二维码图片，
       通过 AlertManager 以邮件附件形式发出，便于守护者远程扫码登录。

设计要点：
    - 所有外部命令路径来自 yaml 配置，绝不从邮件等不可信渠道拼接。
    - Windows 下 .bat 必须经 cmd.exe 调起，create_subprocess_exec 比 shell=True 安全。
    - 单实例：通过 asyncio.Lock 防止并发重启风暴。
    - 状态机由 NapcatClient 与 AlertManager 共同维护，本模块只触发 + 通知。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger("AICQ.napcat_supervisor")


class NapcatSupervisor:
    """NapCat 进程监管器：自动重启 + 二维码邮件。

    cfg 字段（来自 alerting.napcat_restart）:
        enabled: bool                  总开关
        command: str                   启动批处理 / 可执行文件路径（必填）
        args: list[str]                额外参数
        cwd: str                       工作目录，空则用 command 所在目录
        stop_command: str              可选：启动前调用的停止脚本路径
        stop_image_names: list[str]    可选：要结束的进程镜像名列表（仅 Windows）。
                                       默认 ["NapCatWinBootMain.exe"]。会通过 WMIC 按可执行文件路径
                                       过滤，只结束 stop_path_filter 匹配的实例，避免误杀同机其他 NapCat。
        stop_path_filter: str          可选：可执行文件路径前缀过滤器（大小写不敏感）。
                                       留空则自动取 cwd 上级目录（适合 NapCat OneKey 布局）。
                                       仅当路径以此开头的进程才会被杀。
        force_kill_by_image_name: bool 可选：路径过滤失败时是否回退到“按镜像名全杀”。
                                       默认 false，避免误杀。
        stop_grace_seconds: int        停止后等多久再拉起新进程（秒），默认 3
        cooldown_seconds: int          两次重启最小间隔（秒），默认 300
        max_attempts_per_hour: int     1 小时滑窗内最大重启次数，默认 4
        recovery_grace_seconds: int    重启后等待心跳恢复的时间（秒），默认 45
        qrcode_globs: list[str]        二维码图片 glob 列表（相对 cwd），
                                       默认 ["**/qrcode*.png", "cache/qrcode*.png"]
    """

    def __init__(self, cfg: dict | None, client: Any = None, alert: Any = None):
        cfg = cfg or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.command: str = str(cfg.get("command", "") or "").strip()
        self.args: list[str] = [str(a) for a in (cfg.get("args") or [])]
        self.cwd: str = str(cfg.get("cwd", "") or "").strip()
        self.stop_command: str = str(cfg.get("stop_command", "") or "").strip()
        self.stop_image_names: list[str] = [
            str(n).strip() for n in (cfg.get("stop_image_names") or ["NapCatWinBootMain.exe"])
            if str(n).strip()
        ]
        self.stop_path_filter: str = str(cfg.get("stop_path_filter", "") or "").strip()
        self.force_kill_by_image_name: bool = bool(cfg.get("force_kill_by_image_name", False))
        self.stop_grace_seconds: int = max(0, int(cfg.get("stop_grace_seconds", 3)))
        self.cooldown_seconds: int = max(30, int(cfg.get("cooldown_seconds", 300)))
        self.max_attempts_per_hour: int = max(1, int(cfg.get("max_attempts_per_hour", 4)))
        self.recovery_grace_seconds: int = max(5, int(cfg.get("recovery_grace_seconds", 45)))
        self.qrcode_globs: list[str] = [
            str(g) for g in (cfg.get("qrcode_globs") or [
                "**/qrcode*.png",
                "cache/qrcode*.png",
            ])
        ]

        self._client = client          # NapcatClient
        self._alert = alert            # AlertManager
        self._lock: asyncio.Lock = asyncio.Lock()
        self._last_attempt_at: float = 0.0
        self._attempt_history: deque[float] = deque(maxlen=64)
        self._inflight: bool = False   # 当前正在执行重启 + 观察流程

    # ── 配置 / 注入 ─────────────────────────────────────────

    def attach(self, client: Any = None, alert: Any = None) -> None:
        """运行时注入依赖（main.py / 热重载使用）。"""
        if client is not None:
            self._client = client
        if alert is not None:
            self._alert = alert

    def is_configured(self) -> bool:
        """是否具备最低运行条件。"""
        return self.enabled and bool(self.command)

    # ── 对外入口 ────────────────────────────────────────────

    def request_restart(self, reason: str) -> None:
        """非阻塞触发重启。重复触发会被节流策略丢弃。"""
        if not self.is_configured():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("无运行中事件循环，重启请求忽略 (reason=%s)", reason)
            return
        loop.create_task(self._run_restart_flow(reason))

    # ── 核心流程 ────────────────────────────────────────────

    async def _run_restart_flow(self, reason: str) -> None:
        async with self._lock:
            if self._inflight:
                logger.debug("已有重启流程在跑，跳过 (reason=%s)", reason)
                return

            now = time.monotonic()
            # 冷却期检查
            if now - self._last_attempt_at < self.cooldown_seconds:
                wait = self.cooldown_seconds - (now - self._last_attempt_at)
                logger.info("重启冷却中（剩余 %.0fs），跳过 (reason=%s)", wait, reason)
                return
            # 1 小时滑窗熔断
            cutoff = now - 3600
            while self._attempt_history and self._attempt_history[0] < cutoff:
                self._attempt_history.popleft()
            if len(self._attempt_history) >= self.max_attempts_per_hour:
                logger.warning(
                    "1 小时内已重启 %d 次，触发熔断，跳过 (reason=%s)",
                    len(self._attempt_history), reason,
                )
                return

            self._last_attempt_at = now
            self._attempt_history.append(now)
            self._inflight = True

        restart_wall_time = time.time()
        try:
            await self._stop_existing()
            ok = await self._launch()
            if not ok:
                return
            await self._wait_and_followup(reason, restart_wall_time)
        finally:
            self._inflight = False

    async def _stop_existing(self) -> None:
        """启动新进程前先结束旧的 NapCat，避免端口占用。

        优先调用 stop_command；同时如配置了 stop_image_names，在 Windows 上 taskkill /F /IM 。
        所有运作都是“尽力而为”，失败只警告，不阅读本次重启流程。
        """
        # 1) 用户自定义 stop 脚本
        if self.stop_command:
            stop_path = self.stop_command
            if not os.path.isabs(stop_path):
                stop_path = str(Path(stop_path).resolve())
            if Path(stop_path).is_file():
                cwd = self.cwd or str(Path(stop_path).parent)
                is_bat = stop_path.lower().endswith((".bat", ".cmd"))
                if sys.platform == "win32" and is_bat:
                    argv = ["cmd.exe", "/c", stop_path]
                else:
                    argv = [stop_path]
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *argv, cwd=cwd,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=15.0)
                        logger.info("stop_command 执行完成 rc=%s", proc.returncode)
                    except asyncio.TimeoutError:
                        logger.warning("stop_command 执行超时，强制终止")
                        proc.kill()
                except (OSError, ValueError) as e:
                    logger.warning("调用 stop_command 失败: %s", e)
            else:
                logger.warning("stop_command 不存在或不是文件，跳过: %s", stop_path)

        # 2) Windows 按路径过滤后 taskkill /F /T /PID
        if self.stop_image_names and sys.platform == "win32":
            path_filter = self._compute_path_filter()
            for name in self.stop_image_names:
                pids = await self._find_pids_by_image_and_path(name, path_filter)
                if pids:
                    for pid in pids:
                        await self._taskkill_pid(pid, name)
                elif path_filter and self.force_kill_by_image_name:
                    logger.warning(
                        "未找到路径匹配 %s 的 %s，回退为 /IM 全杀（可能误伤其他实例）",
                        path_filter, name,
                    )
                    await self._taskkill_image(name)
                elif not path_filter:
                    # 未提供任何过滤条件、也拿不到 cwd 默认值 → 退回镜像名全杀
                    logger.warning("无路径过滤器，按镜像名结束 %s（可能误伤同机实例）", name)
                    await self._taskkill_image(name)
                else:
                    logger.info("未找到路径匹配 %s 的 %s，跳过", path_filter, name)
        elif self.stop_image_names:
            logger.debug("当前非 Windows 环境，忽略 stop_image_names")

        # 3) 给内核释放端口 / 句柄一点时间
        if self.stop_grace_seconds > 0 and (self.stop_command or self.stop_image_names):
            await asyncio.sleep(self.stop_grace_seconds)

    async def _launch(self) -> bool:
        """拉起子进程，不等待其结束（NapCat 是常驻进程）。"""
        cmd_path = self.command
        if not os.path.isabs(cmd_path):
            cmd_path = str(Path(cmd_path).resolve())
        if not Path(cmd_path).exists():
            logger.error("napcat_restart.command 不存在: %s", cmd_path)
            return False

        cwd = self.cwd or str(Path(cmd_path).parent)
        if not Path(cwd).is_dir():
            logger.error("napcat_restart.cwd 不是有效目录: %s", cwd)
            return False

        # Windows .bat 必须由 cmd.exe 启动；其它平台按可执行文件直接调用
        is_bat = cmd_path.lower().endswith((".bat", ".cmd"))
        if sys.platform == "win32" and is_bat:
            argv = ["cmd.exe", "/c", cmd_path, *self.args]
        else:
            argv = [cmd_path, *self.args]

        try:
            # detach：父进程退出后 NapCat 仍存活
            kwargs: dict[str, Any] = {"cwd": cwd}
            if sys.platform == "win32":
                # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
                kwargs["creationflags"] = 0x00000200 | 0x00000008
            else:
                kwargs["start_new_session"] = True
            proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
            logger.info(
                "已拉起 NapCat 启动脚本 pid=%s argv=%s cwd=%s",
                proc.pid, argv, cwd,
            )
            return True
        except (OSError, ValueError) as e:
            logger.exception("拉起 NapCat 启动脚本失败: %s", e)
            return False

    async def _wait_and_followup(self, reason: str, restart_wall_time: float) -> None:
        """等待心跳恢复；若超时则寻找二维码并发邮件。"""
        deadline = time.monotonic() + self.recovery_grace_seconds
        # 每 2s 轮询一次
        while time.monotonic() < deadline:
            await asyncio.sleep(2.0)
            if self._client_is_alive():
                logger.info(
                    "重启成功：心跳已恢复（耗时约 %.0fs），等待 client 自身的恢复邮件流程",
                    self.recovery_grace_seconds - (deadline - time.monotonic()),
                )
                # 心跳恢复路径会自动触发 alert.notify_recover()，无需此处再发
                return

        logger.warning(
            "重启后 %ds 内心跳未恢复，尝试发送二维码邮件 (reason=%s)",
            self.recovery_grace_seconds, reason,
        )
        if self._alert is None:
            return

        qr_path = await asyncio.to_thread(self._find_latest_qrcode, restart_wall_time)
        if qr_path is None:
            logger.info("未找到重启后生成的二维码文件，跳过二维码邮件")
            try:
                await self._alert.notify_disconnect_followup(
                    f"自动重启已执行但未恢复，且未发现二维码文件。原因: {reason}"
                )
            except AttributeError:
                # 老版 AlertManager 无此方法时静默
                pass
            return

        try:
            await self._alert.notify_qrcode(
                reason, qr_path,
                recovery_hint=f"等待 {self.recovery_grace_seconds}s 内",
            )
        except Exception:
            logger.exception("发送二维码邮件失败")

    # ── 工具 ─────────────────────────────────────────────────────────

    def _compute_path_filter(self) -> str:
        """计算 stop 阶段的路径过滤前缀。统一用反斜杠、全小写。"""
        raw = self.stop_path_filter
        if not raw:
            # 默认取 cwd 的上一级目录，NapCat OneKey 中 cwd 是 bootmain，
            # 上一级才是 NapCat 根目录（以该路径为前缀能包含 NapCat.*.Shell\\QQ.exe）
            base = self.cwd or str(Path(self.command).parent if self.command else "")
            if base:
                raw = str(Path(base).resolve().parent)
        if not raw:
            return ""
        return str(raw).replace("/", "\\").rstrip("\\").lower()

    async def _find_pids_by_image_and_path(self, image_name: str, path_filter: str) -> list[int]:
        """用 WMIC 查出指定镜像且 ExecutablePath 以 path_filter 开头的 PID 列表。

        未提供 path_filter 时返回空列表（调用者需自行处理回退逻辑）。
        """
        if not path_filter:
            return []
        # WMIC 双引号转义麻烦，这里只按镜像名查，路径过滤在 Python 侧做
        argv = [
            "wmic.exe", "process", "where", f"name='{image_name}'",
            "get", "ProcessId,ExecutablePath", "/format:csv",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning("wmic 查询 %s 超时", image_name)
                return []
        except (OSError, ValueError) as e:
            logger.warning("wmic 调用异常: %s", e)
            return []

        pids: list[int] = []
        for line in stdout.decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.lower().startswith("node,"):
                continue
            # CSV 格式: Node,ExecutablePath,ProcessId
            parts = line.split(",")
            if len(parts) < 3:
                continue
            exe_path = parts[1].strip().replace("/", "\\").lower()
            pid_str = parts[2].strip()
            if not exe_path or not pid_str.isdigit():
                continue
            if exe_path.startswith(path_filter):
                pids.append(int(pid_str))
        return pids

    async def _taskkill_pid(self, pid: int, image_hint: str = "") -> None:
        argv = ["taskkill.exe", "/F", "/T", "/PID", str(pid)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning("taskkill /PID %s 超时", pid)
                return
            if proc.returncode == 0:
                logger.info("已结束 %s pid=%s", image_hint or "process", pid)
            else:
                logger.debug("taskkill /PID %s rc=%s", pid, proc.returncode)
        except (OSError, ValueError) as e:
            logger.warning("taskkill /PID %s 调用异常: %s", pid, e)

    async def _taskkill_image(self, name: str) -> None:
        argv = ["taskkill.exe", "/F", "/T", "/IM", name]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning("taskkill /IM %s 超时", name)
                return
            if proc.returncode == 0:
                logger.info("taskkill /IM 已结束 %s", name)
            else:
                logger.debug("taskkill /IM %s rc=%s", name, proc.returncode)
        except (OSError, ValueError) as e:
            logger.warning("taskkill /IM %s 调用异常: %s", name, e)

    def _client_is_alive(self) -> bool:
        c = self._client
        if c is None:
            return False
        try:
            connected = bool(getattr(c, "connected", False))
            stale = bool(getattr(c, "_heartbeat_stale", False))
            return connected and not stale
        except Exception:
            return False

    def _find_latest_qrcode(self, since_wall_time: float) -> Path | None:
        """寻找 mtime 大于 since_wall_time 的最新二维码图片。

        匹配规则：
          - glob 条目以盘符 (X:) 或 / 开头视为绝对路径，使用 glob.glob 处理；
          - 否则在 [cwd, cwd.parent] 两级范围内做相对 glob，
            兼容 NapCat OneKey 那种 cwd=bootmain、二维码在兄弟目录的布局。
        """
        import glob as _glob_mod

        cwd = Path(self.cwd or Path(self.command).parent)
        if not cwd.is_dir():
            return None
        search_bases: list[Path] = [cwd]
        try:
            parent = cwd.resolve().parent
            if parent != cwd and parent.is_dir():
                search_bases.append(parent)
        except OSError:
            pass

        candidates: list[tuple[float, Path]] = []

        def _consider(path_obj: Path) -> None:
            if not path_obj.is_file():
                return
            try:
                mtime = path_obj.stat().st_mtime
            except OSError:
                return
            if mtime + 1.0 < since_wall_time:
                return
            candidates.append((mtime, path_obj))

        for pattern in self.qrcode_globs:
            pattern_norm = pattern.replace("\\", "/")
            is_abs = (
                len(pattern_norm) >= 2 and pattern_norm[1] == ":"  # X:
            ) or pattern_norm.startswith("/")
            try:
                if is_abs:
                    for matched in _glob_mod.glob(pattern_norm, recursive=True):
                        _consider(Path(matched))
                else:
                    for base in search_bases:
                        for p in base.glob(pattern):
                            _consider(p)
            except (OSError, ValueError):
                continue

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
