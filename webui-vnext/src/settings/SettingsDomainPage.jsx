import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CircleAlert,
  CircleCheck,
  KeyRound,
  LoaderCircle,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  ShieldCheck,
  Trash2,
} from "lucide-react";
import {
  loadSecurityStatus,
  loadSettingsDomain,
  replaceSecurityPassword,
  saveSettingsDomain,
} from "../api/settingsApi.js";

const DOMAIN_COPY = {
  providers: {
    group: "模型与推理",
    title: "模型供应商",
    description: "管理供应商端点与凭据引用。API Key 只显示配置状态，不会回传到浏览器。",
  },
  "main-model": {
    group: "模型与推理",
    title: "主模型",
    description: "设置 Core 默认使用的模型与生成预算。保存只修改这一领域。",
  },
  "specialized-models": {
    group: "模型与推理",
    title: "专用模型",
    description: "按用途分配工具守门、压缩、视觉与慢思考模型。",
  },
  persona: {
    group: "记忆与身份",
    title: "角色与身份",
    description: "分别管理身份字段、Persona 正文与 QQ 社交表达偏好。",
  },
  "qq-adapter": {
    group: "接入与表达",
    title: "QQ / Adapter",
    description: "配置 QQ 接入方式、反向 WebSocket 与会话恢复策略。",
  },
  tts: {
    group: "接入与表达",
    title: "TTS",
    description: "配置语音服务连接、并发预算与独立 Secret。",
  },
  services: {
    group: "工具与通知",
    title: "外部服务",
    description: "管理搜索、浏览器与天气服务；凭据不会进入页面状态。",
  },
  alerts: {
    group: "工具与通知",
    title: "告警与邮件",
    description: "分别保存告警策略、SMTP、IMAP 与邮件远程指令配置。",
  },
  advanced: {
    group: "运行与数据",
    title: "网络与高级",
    description: "管理代理 Secret 与低频兼容选项；保留、替换和清除都需要显式选择。",
  },
};

const FIELD_GROUPS = {
  "main-model": [
    {
      title: "模型绑定",
      description: "供应商与模型 ID 必须同时有效。",
      fields: [
        ["provider", "供应商", "Core 默认使用的供应商。", "select", "providers"],
        ["model", "模型 ID", "调用接口时发送的模型标识。", "text"],
        ["model_name", "显示名称", "用于日志和界面识别。", "text"],
      ],
    },
    {
      title: "生成预算",
      description: "这些参数会在重启 Core 后完全应用。",
      fields: [
        ["generation.temperature", "Temperature", "0–2，越高越发散。", "number", null, { min: 0, max: 2, step: 0.05 }],
        ["generation.max_output_tokens", "最大输出 Token", "单次生成的硬上限。", "number", null, { min: 64, max: 262144 }],
        ["generation.enable_thinking", "思考模式", "允许模型使用思考能力。", "boolean"],
        ["max_calls_per_minute", "每分钟最大调用", "全局速率预算。", "number", null, { min: 1, max: 10000 }],
      ],
    },
  ],
  "specialized-models": [
    {
      title: "工具执行守门",
      description: "在工具执行前进行独立判断。关闭后可保留绑定。",
      fields: [
        ["tool_execution_guard.enabled", "启用工具守门", "只影响工具执行前检查。", "boolean"],
        ["tool_execution_guard.provider", "供应商", "守门模型供应商。", "select", "providers"],
        ["tool_execution_guard.model", "模型 ID", "守门模型标识。", "text"],
        ["tool_execution_guard.generation.temperature", "Temperature", "守门判断随机性。", "number", null, { min: 0, max: 2, step: 0.05 }],
        ["tool_execution_guard.generation.max_output_tokens", "最大输出 Token", "守门响应预算。", "number", null, { min: 64, max: 262144 }],
      ],
    },
    {
      title: "上下文压缩",
      description: "长上下文压缩使用独立模型，必须保持有效绑定。",
      fields: [
        ["cognition_compression.provider", "供应商", "压缩模型供应商。", "select", "providers"],
        ["cognition_compression.model", "模型 ID", "压缩模型标识。", "text"],
        ["cognition_compression.generation.temperature", "Temperature", "压缩生成随机性。", "number", null, { min: 0, max: 2, step: 0.05 }],
        ["cognition_compression.generation.max_output_tokens", "最大输出 Token", "压缩输出预算。", "number", null, { min: 256, max: 262144 }],
      ],
    },
    {
      title: "视觉与慢思考",
      description: "仅在对应能力开启时要求完整模型绑定。",
      fields: [
        ["vision_bridge.enabled", "启用 Vision Bridge", "为图片理解使用独立模型。", "boolean"],
        ["vision_bridge.provider", "视觉供应商", "Vision Bridge 供应商。", "select", "providers"],
        ["vision_bridge.model", "视觉模型 ID", "Vision Bridge 模型。", "text"],
        ["slow_thinking.enabled", "启用慢思考", "允许显式深度思考流程。", "boolean"],
        ["slow_thinking.provider", "慢思考供应商", "慢思考模型供应商。", "select", "providers"],
        ["slow_thinking.model", "慢思考模型 ID", "慢思考模型标识。", "text"],
        ["slow_thinking.generation.temperature", "慢思考 Temperature", "慢思考生成随机性。", "number", null, { min: 0, max: 2, step: 0.05 }],
        ["slow_thinking.generation.max_output_tokens", "慢思考 Token", "慢思考输出预算。", "number", null, { min: 64, max: 262144 }],
      ],
    },
  ],
  persona: [
    {
      title: "身份",
      description: "身份字段和 Persona 分开持久化，失败状态互不混淆。",
      fields: [
        ["self_name", "自身名称", "Core 对外使用的名称。", "text"],
        ["guardian.name", "监护人名称", "监护人在上下文中的显示名称。", "text"],
        ["guardian.id", "监护人 ID", "平台侧稳定标识。", "text"],
        ["timezone", "时区", "IANA 时区，例如 Asia/Shanghai。", "text"],
      ],
    },
    {
      title: "角色正文",
      description: "长文本不会存入 LocalStorage。",
      fields: [
        ["persona", "Persona", "角色核心设定。", "textarea", null, { rows: 12 }],
        ["qq_social_style", "QQ 社交风格", "QQ 场景下的表达偏好。", "textarea", null, { rows: 8 }],
      ],
    },
  ],
  "qq-adapter": [
    {
      title: "连接",
      description: "保存后需要 Core 重启，避免出现已保存但误报已应用。",
      fields: [
        ["enabled", "启用 QQ 接入", "允许 Core 接收 QQ 会话。", "boolean"],
        ["adapter.type", "Adapter 类型", "自动检测、NapCat 或 LLOneBot。", "select", "adapter_types"],
        ["adapter.name", "Adapter 名称", "便于区分部署实例。", "text"],
        ["adapter.debug_only", "仅调试", "限制为调试链路。", "boolean"],
        ["adapter.reverse_ws.host", "反向 WS 主机", "监听地址。", "text"],
        ["adapter.reverse_ws.port", "反向 WS 端口", "1–65535。", "number", null, { min: 1, max: 65535 }],
      ],
    },
    {
      title: "访问与恢复",
      description: "保留现有名单，仅调整领域开关。",
      fields: [
        ["access.whitelist.enabled", "启用白名单", "只响应允许的用户与群组。", "boolean"],
        ["attention.respond_to_self_name", "响应自身名称", "检测到名称时提升注意。", "boolean"],
        ["recovery.enabled", "恢复会话", "启动后恢复近期会话。", "boolean"],
        ["recovery.backfill_history", "补齐历史消息", "恢复时补齐缺失上下文。", "boolean"],
      ],
    },
  ],
  tts: [
    {
      title: "语音服务",
      description: "Secret 与普通连接参数独立提交。",
      fields: [
        ["enabled", "启用 TTS", "允许 Core 生成语音。", "boolean"],
        ["host", "服务主机", "TTS 服务监听地址。", "text"],
        ["port", "服务端口", "1–65535。", "number", null, { min: 1, max: 65535 }],
        ["max_concurrent_tasks_per_plugin", "每插件并发", "限制单插件并发任务。", "number", null, { min: 1, max: 128 }],
      ],
    },
  ],
  services: [
    {
      title: "搜索",
      description: "搜索端点与 API Key 分开管理。",
      fields: [
        ["web_search.searxng.enabled", "启用 SearXNG", "使用自托管搜索端点。", "boolean"],
        ["web_search.searxng.base_url", "SearXNG 地址", "完整 HTTP(S) 地址。", "text"],
        ["web_search.searxng.language", "搜索语言", "例如 zh-CN。", "text"],
        ["web_search.searxng.safesearch", "安全搜索", "0 关闭、1 中等、2 严格。", "select", "safesearch"],
      ],
    },
    {
      title: "浏览器与天气",
      description: "Profile 目录由服务端解析，页面不持久化本地路径副本。",
      fields: [
        ["browser_control.profile_dir", "浏览器 Profile", "持久化登录态目录。", "text"],
        ["browser_control.multimodal_image_limit", "图像预算", "一次浏览器观察最多携带的图像数。", "number", null, { min: 0, max: 64 }],
        ["browser_control.annotate_screenshots", "截图标注", "在截图上绘制定位标注。", "boolean"],
        ["service_env.QWEATHER_API_HOST", "天气 API Host", "可选的天气服务地址。", "text"],
      ],
    },
  ],
  alerts: [
    {
      title: "告警策略",
      description: "保存配置不会自动发送测试邮件。",
      fields: [
        ["alerting.enabled", "启用告警", "监控 Core 心跳。", "boolean"],
        ["alerting.heartbeat_timeout", "心跳超时（秒）", "超过后进入告警流程。", "number", null, { min: 30, max: 86400 }],
        ["alerting.cooldown", "冷却时间（秒）", "避免重复告警。", "number", null, { min: 0, max: 604800 }],
        ["alerting.subject_prefix", "主题前缀", "邮件标题的固定前缀。", "text"],
        ["alerting.email_control.enabled", "邮件远程指令", "允许受控的邮件命令。", "boolean"],
        ["alerting.email_control.poll_interval", "轮询间隔（秒）", "10–600。", "number", null, { min: 10, max: 600 }],
        ["alerting.email_control.token_ttl_seconds", "Token 有效期（秒）", "远程指令握手有效期。", "number", null, { min: 60, max: 604800 }],
        ["alerting.email_control.reuse_smtp_credentials", "复用 SMTP 凭据", "IMAP 未单独配置时复用。", "boolean"],
      ],
    },
    {
      title: "SMTP",
      description: "密码在下方 Secret 区域管理。",
      fields: [
        ["smtp.AICQ_SMTP_HOST", "SMTP 主机", "发信服务器地址。", "text"],
        ["smtp.AICQ_SMTP_PORT", "SMTP 端口", "通常为 465 或 587。", "text"],
        ["smtp.AICQ_SMTP_USE_SSL", "使用 SSL", "SMTP 加密方式。", "select", "boolean_text"],
        ["smtp.AICQ_SMTP_USER", "SMTP 用户", "登录用户名。", "text"],
        ["smtp.AICQ_SMTP_SENDER", "发件人", "告警邮件 From。", "text"],
        ["smtp.AICQ_SMTP_RECIPIENTS", "收件人", "多个地址按现有后端格式填写。", "text"],
      ],
    },
    {
      title: "IMAP",
      description: "远程指令收信配置。",
      fields: [
        ["imap.AICQ_IMAP_HOST", "IMAP 主机", "收信服务器地址。", "text"],
        ["imap.AICQ_IMAP_PORT", "IMAP 端口", "通常为 993。", "text"],
        ["imap.AICQ_IMAP_USE_SSL", "使用 SSL", "IMAP 加密方式。", "select", "boolean_text"],
        ["imap.AICQ_IMAP_USER", "IMAP 用户", "登录用户名。", "text"],
      ],
    },
  ],
  advanced: [
    {
      title: "兼容选项",
      description: "只修改明确列出的低频行为。",
      fields: [
        ["tools.send_message.message_shape", "消息发送形态", "array 为多条消息，single 为单条 segments。", "select", "message_shape"],
      ],
    },
  ],
};

const FIXED_SECRET_COPY = {
  tts: [["secret_token", "TTS Secret Token", "TTS 服务认证令牌。"]],
  services: [
    ["tavily_api_key", "Tavily API Key", "外部搜索服务凭据。"],
    ["weather_api_key", "天气 API Key", "天气服务凭据。"],
    ["browser_proxy", "浏览器代理", "浏览器专用 HTTP(S) 上游代理；保存后重启 Core 生效。"],
  ],
  alerts: [
    ["smtp_password", "SMTP 密码", "发信账户密码或授权码。"],
    ["imap_password", "IMAP 密码", "收信账户密码或授权码。"],
  ],
  advanced: [
    ["openai_proxy", "OpenAI 代理", "可包含认证信息，因此按 Secret 管理。"],
    ["tavily_proxy", "Tavily 代理", "可包含认证信息，因此按 Secret 管理。"],
  ],
};

const OPTION_SETS = {
  safesearch: [
    { id: 0, label: "0 · 关闭" },
    { id: 1, label: "1 · 中等" },
    { id: 2, label: "2 · 严格" },
  ],
  boolean_text: [
    { id: "true", label: "是" },
    { id: "false", label: "否" },
  ],
  message_shape: [
    { id: "array", label: "array · 多条消息" },
    { id: "single", label: "single · 单条 segments" },
  ],
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function same(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function getPath(value, path) {
  return path.split(".").reduce((current, key) => current?.[key], value);
}

function setPath(value, path, next) {
  const copy = clone(value);
  const parts = path.split(".");
  let current = copy;
  parts.slice(0, -1).forEach((part) => {
    if (!current[part] || typeof current[part] !== "object") current[part] = {};
    current = current[part];
  });
  current[parts.at(-1)] = next;
  return copy;
}

function initialSecretDrafts(secrets = {}) {
  return Object.fromEntries(
    Object.keys(secrets).map((secretId) => [secretId, { command: "keep", value: "" }]),
  );
}

function secretDraftsDirty(drafts) {
  return Object.values(drafts).some((draft) => draft.command !== "keep");
}

function optionList(source, options) {
  return options?.[source] || OPTION_SETS[source] || [];
}

function ConfigField({ field, value, options, onChange }) {
  const [path, label, hint, type, optionSource, attributes = {}] = field;
  const id = `field-${path.replaceAll(".", "-")}`;

  if (type === "boolean") {
    return (
      <div className="form-row settings-domain-row">
        <span className="form-label"><strong>{label}</strong><small>{hint}</small></span>
        <span className="form-control">
          <label className="domain-toggle" htmlFor={id}>
            <input
              id={id}
              type="checkbox"
              checked={Boolean(value)}
              onChange={(event) => onChange(event.target.checked)}
            />
            <i aria-hidden="true" />
            <span>{value ? "已启用" : "已关闭"}</span>
          </label>
        </span>
      </div>
    );
  }

  return (
    <label className="form-row settings-domain-row" htmlFor={id}>
      <span className="form-label"><strong>{label}</strong><small>{hint}</small></span>
      <span className="form-control">
        {type === "select" ? (
          <select
            id={id}
            value={value ?? ""}
            onChange={(event) => {
              const selected = optionList(optionSource, options).find(
                (item) => String(item.id) === event.target.value,
              );
              onChange(selected?.id ?? event.target.value);
            }}
          >
            {optionList(optionSource, options).map((item) => (
              <option key={String(item.id)} value={item.id}>{item.label}</option>
            ))}
          </select>
        ) : type === "textarea" ? (
          <textarea
            id={id}
            rows={attributes.rows || 6}
            value={value ?? ""}
            onChange={(event) => onChange(event.target.value)}
          />
        ) : (
          <input
            id={id}
            type={type === "number" ? "number" : "text"}
            value={value ?? ""}
            min={attributes.min}
            max={attributes.max}
            step={attributes.step}
            onChange={(event) => onChange(
              type === "number" && event.target.value !== ""
                ? Number(event.target.value)
                : event.target.value,
            )}
          />
        )}
      </span>
    </label>
  );
}

function SecretEditor({ secretId, label, hint, state, draft, onChange }) {
  const command = draft?.command || "keep";
  return (
    <div className={`secret-editor ${command === "clear" ? "will-clear" : ""}`}>
      <div className="secret-editor-copy">
        <span className="secret-icon"><KeyRound size={16} /></span>
        <span>
          <strong>{label}</strong>
          <small>{hint}</small>
        </span>
      </div>
      <div className="secret-editor-controls">
        <span className={`secret-state ${state?.configured ? "configured" : "empty"}`}>
          {state?.configured ? `已配置 · ${state.masked_hint}` : "尚未配置"}
        </span>
        <select
          aria-label={`${label}操作`}
          value={command}
          onChange={(event) => onChange({ command: event.target.value, value: "" })}
        >
          <option value="keep">保留</option>
          <option value="replace">替换</option>
          <option value="clear">清除</option>
        </select>
        {command === "replace" && (
          <input
            type="password"
            value={draft?.value || ""}
            autoComplete="new-password"
            placeholder="输入新值"
            aria-label={`${label}新值`}
            onChange={(event) => onChange({ command, value: event.target.value })}
          />
        )}
        {command === "clear" && <small className="secret-clear-note">保存后清除，无法撤销。</small>}
      </div>
      <input type="hidden" value={secretId} readOnly />
    </div>
  );
}

function ProviderEditor({ draft, secrets, secretDrafts, onDraft, onSecretDrafts }) {
  const providers = draft.model_providers || {};

  const updateProvider = (providerId, key, value) => {
    onDraft({
      ...draft,
      model_providers: {
        ...providers,
        [providerId]: { ...providers[providerId], [key]: value },
      },
    });
  };

  const removeProvider = (providerId) => {
    if (Object.keys(providers).length <= 1) return;
    const nextProviders = { ...providers };
    delete nextProviders[providerId];
    onDraft({ ...draft, model_providers: nextProviders });
    const nextSecrets = { ...secretDrafts };
    delete nextSecrets[`provider_api_key::${providerId}`];
    onSecretDrafts(nextSecrets);
  };

  const addProvider = () => {
    let providerId = `custom_${Date.now()}`;
    while (Object.hasOwn(providers, providerId)) providerId = `${providerId}_2`;
    onDraft({
      ...draft,
      model_providers: {
        ...providers,
        [providerId]: {
          name: "新供应商",
          base_url: "https://api.example.com/v1",
          requires_api_key: true,
          supports_response_format: true,
          thinking_control: "enable_thinking",
          supports_enable_thinking: true,
          supports_assistant_prefill: true,
        },
      },
    });
    onSecretDrafts({
      ...secretDrafts,
      [`provider_api_key::${providerId}`]: { command: "keep", value: "" },
    });
  };

  return (
    <section className="form-section provider-section">
      <div className="form-section-header provider-section-header">
        <div><h3>供应商列表</h3><p>端点与 Key 分离保存；删除操作在保存前可撤销。</p></div>
        <button className="secondary-button" type="button" onClick={addProvider}>
          <Plus size={16} /> 添加供应商
        </button>
      </div>
      <div className="provider-card-list">
        {Object.entries(providers).map(([providerId, provider]) => {
          const secretId = `provider_api_key::${providerId}`;
          return (
            <article className="provider-card" key={providerId}>
              <div className="provider-card-head">
                <div><span>PROVIDER</span><strong>{provider.name || providerId}</strong><code>{providerId}</code></div>
                <button
                  className="icon-text-button danger"
                  type="button"
                  disabled={Object.keys(providers).length <= 1}
                  onClick={() => removeProvider(providerId)}
                >
                  <Trash2 size={15} /> 移除
                </button>
              </div>
              <div className="provider-fields">
                <label><span>显示名称</span><input value={provider.name || ""} onChange={(event) => updateProvider(providerId, "name", event.target.value)} /></label>
                <label><span>API 地址</span><input value={provider.base_url || ""} onChange={(event) => updateProvider(providerId, "base_url", event.target.value)} /></label>
                <label className="provider-check"><input type="checkbox" checked={Boolean(provider.requires_api_key)} onChange={(event) => updateProvider(providerId, "requires_api_key", event.target.checked)} /><span>需要 API Key</span></label>
              </div>
              <SecretEditor
                secretId={secretId}
                label="API Key"
                hint="浏览器只知道是否已配置。"
                state={secrets[secretId]}
                draft={secretDrafts[secretId]}
                onChange={(next) => onSecretDrafts({ ...secretDrafts, [secretId]: next })}
              />
            </article>
          );
        })}
      </div>
    </section>
  );
}

function LoadingState({ label }) {
  return <div className="settings-domain-state" role="status"><LoaderCircle className="spin" size={20} /> {label}</div>;
}

function SecuritySettingsPage({ onToast, onDirtyChange }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [saving, setSaving] = useState(false);
  const dirty = Boolean(password || confirmation);

  useEffect(() => onDirtyChange?.(dirty), [dirty, onDirtyChange]);

  useEffect(() => {
    const controller = new AbortController();
    loadSecurityStatus({ signal: controller.signal })
      .then(setStatus)
      .catch((requestError) => {
        if (requestError?.name !== "AbortError") {
          setError(requestError?.message || "安全状态加载失败");
        }
      });
    return () => controller.abort();
  }, []);

  const save = async () => {
    if (password.length < 6) {
      setError("密码至少需要 6 位");
      return;
    }
    if (password !== confirmation) {
      setError("两次输入的密码不一致");
      return;
    }
    setSaving(true);
    setError("");
    try {
      await replaceSecurityPassword(password);
      setPassword("");
      setConfirmation("");
      setStatus(await loadSecurityStatus());
      onToast("面板密码已更新");
    } catch (requestError) {
      setError(requestError?.message || "密码更新失败");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-form-page narrow-settings-page">
      <div className="settings-page-header">
        <div>
          <div className="breadcrumb">界面与安全 / 面板安全</div>
          <h2>面板安全</h2>
          <p>使用现有 Session 认证管理访问密码。密码不会保存在浏览器状态中。</p>
        </div>
        {status && <span className="domain-revision-chip"><ShieldCheck size={14} /> Session</span>}
      </div>
      {error && <div className="inline-error" role="alert"><CircleAlert size={16} /> {error}</div>}
      {!status ? <LoadingState label="正在加载安全状态" /> : (
        <>
          <section className="security-status-grid" aria-label="安全状态">
            <article><span>密码保护</span><strong>{status.enabled ? "已开启" : "未开启"}</strong></article>
            <article><span>当前会话</span><strong>{status.authenticated ? "已认证" : "未认证"}</strong></article>
            <article><span>会话有效期</span><strong>{status.session_days} 天</strong></article>
          </section>
          {status.external_access_hint && (
            <div className="security-hint"><CircleAlert size={16} /> 当前通过非回环地址访问，建议保持密码保护开启。</div>
          )}
          <section className="form-section">
            <div className="form-section-header"><h3>{status.enabled ? "更换密码" : "设置密码"}</h3><p>提交后当前 Session 保持有效。</p></div>
            <div className="form-rows">
              <label className="form-row settings-domain-row">
                <span className="form-label"><strong>新密码</strong><small>至少 6 位。</small></span>
                <span className="form-control"><input type="password" autoComplete="new-password" value={password} onChange={(event) => setPassword(event.target.value)} /></span>
              </label>
              <label className="form-row settings-domain-row">
                <span className="form-label"><strong>确认密码</strong><small>再次输入以避免误操作。</small></span>
                <span className="form-control"><input type="password" autoComplete="new-password" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></span>
              </label>
            </div>
          </section>
          <div className="sticky-save-bar">
            <div className={dirty ? "dirty" : "synced"}><span />{dirty ? "密码尚未提交" : "安全状态已同步"}</div>
            <div>
              <button className="secondary-button" type="button" disabled={!dirty || saving} onClick={() => { setPassword(""); setConfirmation(""); }}><RotateCcw size={16} /> 重置</button>
              <button className="primary-button" type="button" disabled={!dirty || saving} onClick={save}>{saving ? <LoaderCircle className="spin" size={16} /> : <Save size={16} />}{saving ? "保存中…" : "更新密码"}</button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export function SettingsDomainPage({ domain, onToast, onDirtyChange }) {
  if (domain === "security") {
    return <SecuritySettingsPage onToast={onToast} onDirtyChange={onDirtyChange} />;
  }
  return (
    <DomainSettingsPage
      domain={domain}
      onToast={onToast}
      onDirtyChange={onDirtyChange}
    />
  );
}

function DomainSettingsPage({ domain, onToast, onDirtyChange }) {
  const [snapshot, setSnapshot] = useState(null);
  const [draft, setDraft] = useState(null);
  const [secretDrafts, setSecretDrafts] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [conflict, setConflict] = useState(null);

  const load = useCallback(async (signal) => {
    setLoading(true);
    setError("");
    setConflict(null);
    try {
      const loaded = await loadSettingsDomain(domain, { signal });
      setSnapshot(loaded);
      setDraft(clone(loaded.values));
      setSecretDrafts(initialSecretDrafts(loaded.secrets));
    } catch (requestError) {
      if (requestError?.name !== "AbortError") setError(requestError?.message || "设置加载失败");
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, [domain]);

  useEffect(() => {
    const controller = new AbortController();
    loadSettingsDomain(domain, { signal: controller.signal })
      .then((loaded) => {
        setSnapshot(loaded);
        setDraft(clone(loaded.values));
        setSecretDrafts(initialSecretDrafts(loaded.secrets));
      })
      .catch((requestError) => {
        if (requestError?.name !== "AbortError") {
          setError(requestError?.message || "设置加载失败");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [domain]);

  const dirty = useMemo(
    () => Boolean(snapshot && draft && (!same(snapshot.values, draft) || secretDraftsDirty(secretDrafts))),
    [draft, secretDrafts, snapshot],
  );

  useEffect(() => onDirtyChange?.(dirty), [dirty, onDirtyChange]);

  const copy = DOMAIN_COPY[domain];
  if (!copy) return null;

  const reset = () => {
    if (!snapshot) return;
    setDraft(clone(snapshot.values));
    setSecretDrafts(initialSecretDrafts(snapshot.secrets));
    setConflict(null);
    setError("");
  };

  const acceptLatest = () => {
    if (!conflict) return;
    setSnapshot(conflict);
    setDraft(clone(conflict.values));
    setSecretDrafts(initialSecretDrafts(conflict.secrets));
    setConflict(null);
    setError("");
  };

  const save = async () => {
    if (!snapshot || !draft || !dirty) return;
    setSaving(true);
    setError("");
    setConflict(null);
    try {
      const saved = await saveSettingsDomain(domain, {
        revision: snapshot.revision,
        values: draft,
        secrets: secretDrafts,
      });
      setSnapshot(saved);
      setDraft(clone(saved.values));
      setSecretDrafts(initialSecretDrafts(saved.secrets));
      onToast(saved.restart_required ? "配置已保存，重启 Core 后完全生效" : "配置已保存并应用");
    } catch (requestError) {
      if (requestError?.status === 409 && requestError?.payload?.latest) {
        setConflict(requestError.payload.latest);
      } else {
        setError(requestError?.message || "设置保存失败");
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-form-page">
      <div className="settings-page-header">
        <div>
          <div className="breadcrumb">{copy.group} / {copy.title}</div>
          <h2>{copy.title}</h2>
          <p>{copy.description}</p>
        </div>
        {snapshot && <span className="domain-revision-chip" title={snapshot.revision}><CircleCheck size={14} /> rev {snapshot.revision.slice(0, 7)}</span>}
      </div>

      {dirty && (
        <div className="unsaved-banner" role="status">
          <span><span className="dirty-dot" /> 有尚未保存的领域更改</span>
          <button type="button" onClick={reset}>撤销全部</button>
        </div>
      )}
      {conflict && (
        <div className="settings-conflict" role="alert">
          <CircleAlert size={18} />
          <div><strong>此领域已在别处更新</strong><span>为避免静默覆盖，当前草稿没有写入。加载最新值后再编辑。</span></div>
          <button className="secondary-button" type="button" onClick={acceptLatest}><RefreshCw size={15} /> 加载最新</button>
        </div>
      )}
      {error && <div className="inline-error" role="alert"><CircleAlert size={16} /> {error}</div>}

      {loading ? <LoadingState label={`正在加载${copy.title}`} /> : !draft || !snapshot ? (
        <div className="settings-domain-state">
          <span>无法读取该领域。</span>
          <button className="secondary-button" type="button" onClick={() => load()}><RefreshCw size={16} /> 重试</button>
        </div>
      ) : (
        <>
          {domain === "providers" ? (
            <ProviderEditor
              draft={draft}
              secrets={snapshot.secrets}
              secretDrafts={secretDrafts}
              onDraft={setDraft}
              onSecretDrafts={setSecretDrafts}
            />
          ) : FIELD_GROUPS[domain]?.map((group) => (
            <section className="form-section" key={group.title}>
              <div className="form-section-header"><h3>{group.title}</h3><p>{group.description}</p></div>
              <div className="form-rows">
                {group.fields.map((field) => (
                  <ConfigField
                    key={field[0]}
                    field={field}
                    value={getPath(draft, field[0])}
                    options={snapshot.options}
                    onChange={(value) => setDraft((current) => setPath(current, field[0], value))}
                  />
                ))}
              </div>
            </section>
          ))}

          {FIXED_SECRET_COPY[domain]?.length > 0 && (
            <section className="form-section">
              <div className="form-section-header"><h3>Secret</h3><p>每个 Secret 必须明确选择保留、替换或清除。</p></div>
              <div className="secret-list">
                {FIXED_SECRET_COPY[domain].map(([secretId, label, hint]) => (
                  <SecretEditor
                    key={secretId}
                    secretId={secretId}
                    label={label}
                    hint={hint}
                    state={snapshot.secrets[secretId]}
                    draft={secretDrafts[secretId]}
                    onChange={(next) => setSecretDrafts((current) => ({ ...current, [secretId]: next }))}
                  />
                ))}
              </div>
            </section>
          )}

          {snapshot.warnings?.length > 0 && (
            <div className="settings-warning-note">{snapshot.warnings.join(" ")}</div>
          )}

          <div className="sticky-save-bar">
            <div className={dirty ? "dirty" : "synced"}><span />{dirty ? "领域配置已修改" : "已与服务端 revision 同步"}</div>
            <div>
              <button className="secondary-button" type="button" disabled={!dirty || saving} onClick={reset}><RotateCcw size={16} /> 重置</button>
              <button className="primary-button" type="button" disabled={!dirty || saving || Boolean(conflict)} onClick={save}>{saving ? <LoaderCircle className="spin" size={16} /> : <Save size={16} />}{saving ? "保存中…" : "保存更改"}</button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
