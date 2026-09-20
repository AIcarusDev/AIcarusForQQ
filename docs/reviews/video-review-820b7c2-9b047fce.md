# 视频功能完整审核：820b7c2e → 9b047fce

审核日期：2026-09-19。结论：当前版本存在影响主要使用路径的缺陷，不能视为视频功能已经端到端可用。确认 11 项问题，其中 P1 3 项、P2 8 项；另列未接入下载函数的潜在缺陷，不混同为线上已触发故障。

> 以下审核正文记录的是原始提交 `9b047fce` 的状态及行号。用户随后授权修复，工作区修复状态见文末，不应把历史问题当成修复后仍然存在的问题。

## 范围与证据边界

- 包含 `820b7c2e8fc37c917dfc6469990d0aed202cfaca` 本身，比较 `820b7c2^..9b047fce`，共 6 个提交、22 个文件，按最终版本审查，不重复报告中间版本已被覆盖的行为。
- 当前 HEAD：`9b047fcef497dcd05f59a5fa6191d9e81bb84391`。通过 `git ls-remote origin refs/heads/cmer` 确认远端 cmer 同一哈希。
- 审核前工作区干净。没有修改业务代码、配置、生产数据库，也没有重启应用或向 QQ 发送消息；只新增本报告。
- 源码检查覆盖全部变更文件，并追踪工具发现、工具结果 XML/多模态回传、Workspace 文件转运、媒体身份绑定、配置保存与热更新等实际消费者。
- 使用临时目录、隔离数据库、合成视频、模拟 HTTP/设置保存进行复现。真实 ffmpeg/ffprobe 已执行；真实供应商视频分析、真实 QQ 视频接收下载、浏览器视频下载和页面点击操作未做端到端验证。

## 确认问题

### F01 · P1 · 上下文里的 video_ref 无法触发下载，主要入口不可用

位置：[common.py:55](E:/Aic_forQ/core/src/tools/video/common.py:55)、[segments.py:501](E:/Aic_forQ/core/src/platforms/qq/adapter/segments.py:501)、[video_store.py:42](E:/Aic_forQ/core/src/llm/media/video_store.py:42)。

QQ 接收和浏览器快照生成了 video_ref，三个工具却都只调用 `locate_video()` 查找已经存在的本地文件。全仓搜索 `download_video_for_ref` 只有定义和测试导入，没有生产调用者，也没有工具把 ref 解析回来源 URL 并下载。QQ 的 URL 仅保存在消息 segment 中，传到模型的 XML 只有 ref。

复现：用真实 `build_content_segments()` 构造带 URL 的视频消息，再把生成的 ref 传给真实 `get_video_info.execute()`，返回 `video_processing_failed`，提示无法定位已下载文件。分析和截帧共享同一解析器，具有相同失败条件。

影响：技能中“用户发视频 → get_video_info → analyze_video”的标准流程无法完成。需要持久化引用到来源的映射，在解析器接入按需下载及失效 URL 恢复；不能只新增一个没有调用者的下载函数。浏览器 blob/MSE 来源也需要明确处理能力或返回可执行的限制说明。

### F02 · P1 · path 直接读取 Windows 宿主文件，既不支持 Agent 电脑路径，又越过已有文件边界

位置：[common.py:49](E:/Aic_forQ/core/src/tools/video/common.py:49)、[client.py:184](E:/Aic_forQ/core/src/tools/video/client.py:184)。

项目的 computer 是隔离 Linux 设备，文件位于 `/home/agent`；已有 `view_image` 使用 `WorkspaceService.stage_host_file()` 安全转运。新视频工具把模型传入的字符串直接交给宿主 `Path(...).resolve()`，没有 Workspace 转运或允许目录检查。

复现：本机 `/home/agent/demo.mp4` 被解释成 `E:\home\agent\demo.mp4`，因此 Agent 下载/处理的视频无法交给视频工具。反过来，宿主任意可读文件都能通过 `is_file()`；以临时普通文本文件调用客户端，其字节被编码并传入模拟模型请求，未知后缀还会标成 `video/mp4`。验证未读取真实私密文件，也未发出真实上传。

影响：正常工作流断开，同时新增可把宿主文件传给外部模型的入口。应沿用 Agent-home 转运契约；如确需受控宿主来源，应使用系统提供的引用/明确目录限制，并在上传前检查真实媒体类型。

### F03 · P1 · 视频 URL 绕过已有脱敏，访问凭据进入模型上下文

位置：[world_prompt.py:515](E:/Aic_forQ/core/src/browser/world_prompt.py:515)、[session.py:1578](E:/Aic_forQ/core/src/browser/session.py:1578)。

图片资源通过 `project_source_url()` 投影：即使 full 模式也会去除 URL 用户名密码并遮蔽 token/signature 等敏感参数。新 `<video>` 的 src/poster 直接进行 XML 转义，未执行同样的脱敏。

复现：输入虚构 `https://user:pass@cdn.example/v.mp4?token=FAKE-PRIVATE&signature=FAKE-SIG`，现有投影会去除账号信息、遮蔽参数；新视频 XML 保留全部值。XML escape 只改变语法，不保护 URL 凭据。

影响：签名媒体地址、poster 查询参数或 Basic Auth 信息会进入外部 LLM 请求和 prompt 快照。应在模型投影阶段复用安全 URL 投影规则，原始 URL 留在内部来源记录中。

### F04 · P2 · 任意分析文本未启用 CDATA，破坏 action_response 的 XML 边界

位置：[analyze_video.py:72](E:/Aic_forQ/core/src/tools/video/analyze_video.py:72)。

`analysis` 是外部模型的任意文本，`prompt_used` 也可以包含任意内容，但该工具没有声明 `RESULT_CDATA=True`。实际 `_format_action_response_item_xml()` 默认直接把 JSON 放入 `<result>`，不进行 XML escape；JSON 编码并不会处理 `&`、`<` 或结束标签。仓库工具约定明确要求此类返回声明 CDATA。

复现：分析文本为 `The screen shows A & B < C` 时，实际 flow 序列化后的 XML 不能通过 XML 解析。若返回标签文本，还会产生伪造结构边界。这里没有声称现有运行时一定因 XML 解析抛错；确定影响是模型消费的结构不再可靠。

应声明该结果使用 CDATA，并检验真实返回序列化、持久化恢复后的结构边界。

### F05 · P2 · OpenAI 兼容路径静默丢弃 mode，却返回用户请求的模式

位置：[client.py:220](E:/Aic_forQ/core/src/tools/video/client.py:220)、[analyze_video.py:76](E:/Aic_forQ/core/src/tools/video/analyze_video.py:76)。

`mode` 只进入原生 `_call_google_native()`；兼容分支的函数签名和 payload 均没有这个参数。显式 openai_compatible 或 auto 从原生失败回退时，agentic 请求变成普通视频请求，结果却仍写 `mode: agentic`。

复现：固定文件、prompt、配置，分别以 static/agentic 调用真实客户端并捕获 HTTP JSON，两份 payload 完全相同。

影响：模型按技能说明选择动态检索，却无法知道实际未请求此行为，可能误判长视频检索覆盖和精度。应支持协议实际能力，或明确拒绝/报告降级，并返回实际使用模式。原生字段 `mediaProcessing` 本身是当前有效字段，不应误删。[Google 视频理解文档](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding)

### F06 · P2 · 用 FPS 换算帧号无法保证精准截帧，总帧数估算也被当成事实

位置：[capture_video_frame.py:73](E:/Aic_forQ/core/src/tools/video/capture_video_frame.py:73)、[common.py:152](E:/Aic_forQ/core/src/tools/video/common.py:152)。

frame_index 被转换为 `index / fps` 再按时间 seek；VFR 不满足帧序号和时间线性对应。ffprobe 失败时还无条件猜 30 FPS。返回的 frame_index 是请求值/换算值，并非实际解码序号；缺失 nb_frames 时，duration×fps 的估算也无标记地放进 total_frames。

真实复现：用 ffmpeg 生成首帧停留 1 秒、后续帧约 0.1 秒的 7 帧 FFV1/MKV。请求零基 `frame_index=2`，工具按 25 FPS 算为 0.08 秒，返回绿色帧；通过 `select=eq(n,2)` 精确解码得到蓝色帧。同一文件 get_video_info 返回 `total_frames=39`。

影响：不符合“精准截取某一帧”的描述，短瞬间核对可能得到错误证据。帧号调用应按真实解码索引定位；时间调用应区分请求时间与实际 PTS；估算帧数必须标明性质，无法取得必要数据时不要猜测后报成功。

### F07 · P2 · 在设置页更换供应商，旧显式地址与 Key 仍会覆盖新选择

位置：[routes_settings.py:1017](E:/Aic_forQ/core/src/web/routes_settings.py:1017)、[client.py:76](E:/Aic_forQ/core/src/tools/video/client.py:76)。

配置模板支持显式 base_url/api_key/api_key_env。设置页只有 provider/model 等字段，保存时复制整个旧 video_understanding，保留不可见的覆盖字段。客户端又优先使用这些显式字段。

隔离 Quart 路由复现：旧配置显式指向 `https://old.invalid/v1`，POST 选择新 provider（地址 `https://new.invalid/v1`）后返回 200，但 `get_video_config()` 仍得到旧 URL 和旧的虚构 Key，只有 model 改成新模型。

影响：界面显示供应商已切换，实际视频仍发给旧供应商，或因新模型配旧端点而失败。应明确“显式覆盖”和“继承 provider”两种模式，让 UI 能展示、清除或编辑覆盖项。

### F08 · P2 · 大小限制没有硬上限，也没有核算内联请求体

位置：[routes_settings.py:1028](E:/Aic_forQ/core/src/web/routes_settings.py:1028)、[client.py:121](E:/Aic_forQ/core/src/tools/video/client.py:121)。

UI 和模板宣称 128 MB 硬上限，服务端只把任意数字转 float，客户端直接拿该数值做上限，没有独立 hard cap。配置也不校验有限正数、timeout 范围或 protocol 枚举。

复现：POST `max_size_mb=1024, timeout=-1, protocol=typo` 返回 200 并生效；129 MiB 临时稀疏文件通过 `validate_file_size()`。没有读取或上传该大文件。

此外，仅检查原始文件大小不足以保证 Gemini 官方内联请求可用：18 MiB 文件 Base64 后约 24 MiB，仍会通过模板默认 20 MiB 检查。当前官方文档以总请求大小 20 MB 为内联边界；代码始终 inline，未按目标服务检查编码后的载荷。代理可能有不同限额，不能把代理经验直接套给官方端点。[Google 内联视频限制](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding)

应在保存和运行时校验有效值、落实硬上限，并按目标服务检查实际 JSON 体积。若保持“不走 File API”的设计，应在发送前明确拒绝超限并说明可接受大小。

### F09 · P2 · 同一浏览器视频每次快照都分配新引用

位置：[session.py:1573](E:/Aic_forQ/core/src/browser/session.py:1573)。

world_snapshot 每次遇到 video 就调用持久化 `generate_time_ref()`，没有按 tab/媒体来源缓存。页面和视频未变，引用也不断变化；每次还向共享媒体身份账本写入一个永久预留。

复现：同一 BrowserSession、相同 viewport_visuals 连续快照两次，生成两个不同 video_ref。

影响：跨轮视频身份不稳定、上下文持续变化、账本积累无内容预留；接通下载后还可能造成相同视频重复下载。应为同一媒体保持稳定身份，在来源确实变化时再分配新引用。这与 F01 的“完全未接入下载”是不同的修复点。

### F10 · P2 · 图片注册失败被吞掉，截帧仍宣称成功且返回空 image_ref

位置：[common.py:261](E:/Aic_forQ/core/src/tools/video/common.py:261)、[capture_video_frame.py:90](E:/Aic_forQ/core/src/tools/video/capture_video_frame.py:90)。

注册函数捕获所有异常返回空字符串，调用者仍返回 success。真实注册可能因为数据库/磁盘故障或媒体大小、像素限制拒绝，工具却继续把字节放进一次性多模态附件。

复现：模拟注册返回空值，真实工具结果为 `status=success, image_ref=""`。技能承诺成功后有持久化引用，但下一轮复用、发送、重启恢复均没有可用凭据。

应区分“截帧成功但持久化失败”或把注册失败作为失败返回，避免宣称已经生成可复用引用。

### F11 · P2 · 原生端点规范化会拼出重复版本路径

位置：[client.py:137](E:/Aic_forQ/core/src/tools/video/client.py:137)。

官方域名分支只接受以 v1beta 结尾的路径，其余情况直接追加 `/v1beta`。配置 `https://generativelanguage.googleapis.com/v1` 时，实际生成 `https://generativelanguage.googleapis.com/v1/v1beta`；后续再追加 `/models/...:generateContent`。gemini_native 会直接失败，auto 后续尝试 chat/completions 也不会修复这个原生根路径。

复现：直接调用真实规范化函数，得到上述重复版本路径。这里只验证了 URL 构造错误，未调用线上 API。

应结构化解析 host/path，识别根路径、已带版本路径和兼容路径，避免对任何非 v1beta 路径盲目追加版本。

## 尚未接入的下载函数：修复 F01 时必须同时处理

这些是 `download_video_for_ref()` 的代码级风险。因为当前没有生产调用者，不计入以上线上可达问题数，也不声称已经造成线上下载损坏。

- 缺少下载字节上限：流式写盘降低内存占用，但没有 Content-Length/累计字节限制，不能依赖分析前的文件大小检查保护下载阶段。
- ref 未做安全字符和根目录约束就参与 `target_dir / f"{ref}.mp4"` 拼接。`locate_video()` 的后备路径也绕过共享 locate_media_file 的 safe-ref/inside-root 检查。接线时不能把任意输入直接传入。
- 同一 ref 使用固定 `.mp4.tmp`，没有锁或独占文件，多并发下载可能冲突；最终文件 replace 在身份绑定之前，绑定失败时只清理 tmp，不撤销已经发布的目标文件，下一次 locate 会直接接受它。
- 所有来源均落盘为 `.mp4`，真实 WebM/MOV 会被分析客户端按后缀声明成 MP4，应保存真实容器/MIME，而不是只改文件名。
- 缺少来源失效恢复、浏览器 cookie/Referer 上下文以及 blob/MSE 处理。需要在支持范围和报错中明确能力。

## Prompt、文档与测试一致性

- video 技能文件与 template 一致，命名空间绑定到三个真实工具；未发现工具名拼错或未被 loader 发现的问题。
- 定向 prompt 会作为用户文本发送，空 prompt 不伪造用户问题；系统指令作为 system/systemInstruction 分开发送。这一设计在模拟请求中成立。
- `static`/`agentic` 是真实的原生 API 能力，当前文档也确认 agentic 有模型适用范围；主要缺陷是兼容分支丢参，以及技能没有给出能力限制。[Google 当前模式说明](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding)
- “精准帧号”“成功后持久化”“video_ref 可直接使用”“128 MB 硬限制”等描述必须在修复 F01/F06/F08/F10 后才能成立，不能只调整文案掩盖主要功能缺失。
- 原生响应当前拼接所有 text part，包括可能标记 `thought=true` 的部分；当前请求没有开启 includeThoughts，因此不作为必现问题。若代理返回思考内容，应过滤 thought part，避免把推理过程当分析结论。
- 新测试对固定 system instruction 英文句子做字符串断言，与 TESTING.md 的静态 prompt 测试禁令不一致；应测试 role/结构/参数路由，不固定可编辑文案。
- 新增测试主要验证格式和模拟返回，没有覆盖 ref 到下载到工具、真实 VFR、模式回退、配置覆盖字段、原始 URL 脱敏和任意文本 XML 边界，因此现有测试通过不能证明这些功能已经打通。

## 全文件覆盖矩阵

| 变更面 | 文件 | 审核结果 |
| --- | --- | --- |
| 浏览器采集、世界提示 | src/browser/session.py；src/browser/world_prompt.py | F01、F03、F09；检查视频属性和图片资源相邻链路 |
| 媒体定位与下载 | src/llm/media/media_storage.py；src/llm/media/video_store.py | F01；未接入下载函数风险单列 |
| QQ 消息与 XML | src/platforms/qq/adapter/segments.py；src/platforms/chat/xml_builder.py | ref/URL/时长/大小保留和 XML escaping；F01 |
| 工具声明及执行 | src/tools/video/__init__.py；analyze_video.py；capture_video_frame.py；get_video_info.py；client.py；common.py | F01、F02、F04、F05、F06、F08、F10、F11 |
| 命名空间与技能 | src/tools/namespaces.yaml；src/skills/video/SKILL.md；SKILL.md.template | loader/技能绑定测试通过；核对功能说明与真实能力 |
| 配置与界面 | src/templates/settings.html；src/web/routes_settings.py；templates/config.yaml.template | 检查读取、provider picker、序列化、保存和热更新；F07、F08；未实点页面 |
| 新增/修改测试 | tests/test_qq_segments.py；test_settings_save.py；test_video_store.py；test_video_tool.py | 31 项通过；缺失端到端和负向边界覆盖 |

## 验证结果

两组现有测试合计 **149 passed、1 skipped**；跳过的是宿主无法创建符号链接的媒体测试。仅出现 jieba/pkg_resources 既有弃用警告。

```powershell
python -B -m pytest tests/test_video_tool.py tests/test_video_store.py tests/test_qq_segments.py tests/test_settings_save.py -q -p no:cacheprovider
python -B -m pytest tests/test_tool_namespaces.py tests/test_skill_namespace_binding.py tests/test_browser_world_prompt.py tests/test_browser_session_image_pipeline.py tests/test_browser_image_resources.py tests/test_prompt_xml_builder_cards.py tests/test_media_identity.py tests/test_media_disk_pipeline.py tests/test_media_registry.py tests/test_schema_contract.py -q -p no:cacheprovider
```

另外执行临时复现：真实 QQ segment 到视频引用解析、连续浏览器快照、ffmpeg/ffprobe VFR 截帧与元数据、隔离媒体注册、HTTP payload 模式对比、Quart 设置保存及生效配置、XML 解析和 URL 脱敏比较、原生端点规范化。复现用的数据库、合成媒体及大文件均位于自动清理的临时目录；没有新增持久测试或改动生产数据。

建议修复顺序：先 F01/F02/F03 打通来源并守住文件和凭据边界，再修 F04/F05/F06 保证模型证据可靠，随后完成设置、大小检查、引用稳定性和失败状态。修复后再用真实 QQ 视频、真实网页视频与实际配置供应商做端到端验收。

## 2026-09-19 授权修复结果

11 项已在工作区修复，未提交、未推送、未重启生产服务。

| 原问题 | 修复行为 |
| --- | --- |
| F01 | SQLite 持久化视频来源；QQ/浏览器登记引用；解析器按需下载；旧 QQ 消息引用可从持久化 segment 恢复来源 |
| F02 | path 限定 `/home/agent` Linux 路径，通过现有 WorkspaceService 转运；处理完成/失败均退出临时文件上下文；上传前验证视频流与容器 |
| F03 | 视频 src/poster 复用 URL 脱敏，遵守 hidden/sanitized/full 投影；真实 URL 仅保存在内部来源记录 |
| F04 | 三个视频工具结果均声明 CDATA，任意文本不会变成外层 XML 结构 |
| F05 | agentic 仅走原生协议，不在原生失败时静默退回兼容协议；不支持时明确报错；过滤原生响应 thought 文本 |
| F06 | 帧号直接按解码索引选择；时间调用返回请求时间，不伪造实际帧号；未知总帧数为 null，估算单列 |
| F07 | 设置页明确选择继承供应商或专用连接；继承时清除旧覆盖，专用端点/Key/Key 环境变量可编辑 |
| F08 | 保存及运行时校验有限数值/范围/协议；128 MiB 文件硬上限；检查 Base64 与文本组成的实际请求体大小 |
| F09 | 根据页面、frame 和视频来源保持稳定引用，其他视觉元素改变行序不改变视频身份 |
| F10 | 注册失败返回业务错误，不返回 success 和空引用 |
| F11 | 按 URL 结构规范化原生端点，消除重复版本路径 |

下载接入时一并处理：公网网关及重定向校验、完整下载超时、声明长度和累计字节限额、唯一临时文件、内容身份先绑定再原子发布、并发不覆盖已有文件、失败清理，以及按真实容器保存扩展名。错误不回显原始签名 URL。

验证包括真实 ffmpeg/ffprobe VFR 样本、隔离媒体数据库和 HTTP transport、QQ segment 到下载/读取/截帧/图片复用全链路、旧引用恢复、工作区转运生命周期、同时发起的下载、配置保存与热更新、任意文本 XML 和原生模式/响应路由。新增测试不固定 prompt 文案。

还通过 Playwright 在隔离设置页实际选择并保存两种连接方式，捕获 POST 验证字段，再由独立 Quart 路由测试验证后端应用行为。页面使用虚构供应商与 Key，没有连接真实配置；测试浏览器及临时服务已关闭。捕获的请求位于 `output/playwright/video-settings-posts.jsonl`（本地测试产物）。

修复后最终回归：**211 passed、1 skipped**（宿主不能创建符号链接），全部修改 Python 文件及新增测试的 Ruff 检查通过，`git diff --check` 通过。最终测试范围为原报告的两组测试，加上 `tests/test_video_integration.py`、`tests/test_video_settings.py`、`tests/test_consciousness_flow_xml.py`；使用 `python -B -m pytest ... -q -p no:cacheprovider` 执行。

能力限制已同步到 video 技能及 template：只直接下载可访问的 HTTP(S) 视频；需要浏览器登录态、失效链接、blob/MSE 来源会明确要求重新取得来源，或使用 computer 保存到 `/home/agent` 后传 path。未增加专门的流媒体下载器或供应商 File API。agentic 仍依赖供应商和模型实际支持。没有执行真实 QQ 收发、生产供应商付费推理或生产重启，隔离验证不代表线上已经部署生效。
