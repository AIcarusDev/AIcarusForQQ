"""test_smtp.py — SMTP 告警邮件投递测试脚本

用途：在不依赖 NapCat 的前提下，独立验证 SMTP 配置能否成功发送告警邮件。

使用方式：
    1. 在项目根目录的 .env 文件中加入 SMTP 凭据，例如：
           AICQ_SMTP_HOST=smtp.qq.com
           AICQ_SMTP_PORT=465
           AICQ_SMTP_USE_SSL=true
           AICQ_SMTP_USER=your_account@qq.com
           AICQ_SMTP_PASSWORD=your_authorization_code
           AICQ_SMTP_SENDER=your_account@qq.com
           AICQ_SMTP_RECIPIENTS=you@example.com,another@example.com

    2. 运行：
           python scripts/test_smtp.py

    3. 也可在命令行覆盖收件人：
           python scripts/test_smtp.py --to you@example.com

注意：
    - QQ 邮箱 / 163 等需使用"授权码"（不是登录密码）。
    - 端口 465 → SSL；端口 587 → STARTTLS；端口 25 → 明文（不推荐）。
"""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path


def _load_dotenv(env_path: Path) -> None:
    """极简 .env 读取（避免引入新依赖）。已存在的环境变量不会被覆盖。"""
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def main() -> int:
    parser = argparse.ArgumentParser(description="SMTP 告警邮件投递测试")
    parser.add_argument("--to", help="覆盖收件人（逗号分隔）", default=None)
    parser.add_argument("--subject", help="自定义主题", default=None)
    parser.add_argument("--body", help="自定义正文", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    _load_dotenv(project_root / ".env")

    host = _env("AICQ_SMTP_HOST")
    port = int(_env("AICQ_SMTP_PORT", "465") or "465")
    use_ssl = _bool_env("AICQ_SMTP_USE_SSL", port == 465)
    user = _env("AICQ_SMTP_USER")
    password = _env("AICQ_SMTP_PASSWORD")
    sender = _env("AICQ_SMTP_SENDER") or user
    recipients_raw = args.to or _env("AICQ_SMTP_RECIPIENTS")
    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

    print("=" * 60)
    print("SMTP 配置预览：")
    print(f"  host        : {host}")
    print(f"  port        : {port}")
    print(f"  use_ssl     : {use_ssl}")
    print(f"  username    : {user}")
    print(f"  password    : {'*' * len(password) if password else '(空)'}")
    print(f"  sender      : {sender}")
    print(f"  recipients  : {recipients}")
    print("=" * 60)

    missing = [
        name for name, val in [
            ("AICQ_SMTP_HOST", host),
            ("AICQ_SMTP_USER", user),
            ("AICQ_SMTP_PASSWORD", password),
        ] if not val
    ]
    if missing:
        print(f"[错误] 缺少必填配置: {', '.join(missing)}")
        print("       请在项目根目录 .env 或环境变量中补齐后重试。")
        return 2
    if not recipients:
        print("[错误] 未指定收件人。请设置 AICQ_SMTP_RECIPIENTS 或使用 --to 参数。")
        return 2

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = args.subject or f"[AIcarus 告警-自检] SMTP 通道连通性测试 @ {ts}"
    body = args.body or (
        f"这是一封来自 AIcarusForQQ 的 SMTP 自检邮件。\n\n"
        f"发送时间: {ts}\n"
        f"发件主机: {host}:{port} (SSL={use_ssl})\n"
        f"发件账号: {user}\n\n"
        f"如果你收到了这封邮件，说明 NapCat 掉线告警的邮件通道已就绪。\n"
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    print(f"\n[发送中] 连接 {host}:{port} ...")
    try:
        if use_ssl:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=20) as s:
                s.set_debuglevel(1)
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.set_debuglevel(1)
                s.ehlo()
                if port != 25:
                    s.starttls(context=ssl.create_default_context())
                    s.ehlo()
                s.login(user, password)
                s.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        print(f"\n[失败] SMTP 认证失败: {e}")
        print("       常见原因：使用了登录密码而非授权码 / 未在邮箱后台开启 SMTP 服务。")
        return 1
    except smtplib.SMTPConnectError as e:
        print(f"\n[失败] 无法连接 SMTP 服务器: {e}")
        return 1
    except (smtplib.SMTPException, OSError) as e:
        print(f"\n[失败] 发送过程中出现异常: {type(e).__name__}: {e}")
        return 1

    print("\n[成功] 邮件已成功提交至 SMTP 服务器。")
    print("       请到收件箱（含垃圾邮件文件夹）确认是否收到。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
