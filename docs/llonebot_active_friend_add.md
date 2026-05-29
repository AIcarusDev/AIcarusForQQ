# LLoneBot 主动加 QQ 好友实现记录

日期：2026-05-29

## 结论

基于当前 LLoneBot/QQNT 架构，主动加好友可以做到“服务端生成好友申请”。本次验证中，发起账号 `213628848` 对目标账号 `3975859721` 在 `2026-05-29 22:40` 发出的申请，已经出现在目标账号 QQ 电脑端的“已过滤的通知”里。

这说明 Lagrange 的公开实现方向是可靠的：关键协议是 `OidbSvcTrpcTcp.0x7c2_5`。但结果不等价于“手机端普通通知可见”。目标手机端可能直接忽略过滤类申请，目标电脑端仍可在“已过滤的通知”中看到。

因此，产品层不能把该能力描述为“保证对方手机收到好友申请”。更准确的边界是：

- 能发起 QQ 好友申请，并进入服务端好友申请流。
- 申请可能被 QQ 风控/过滤逻辑放入目标端“已过滤的通知”。
- 发送端返回成功或出现本地 pending 只证明申请已提交/进入请求流，不证明目标手机端普通通知可见。

## 已验证环境

- 发起账号：`213628848`
- 目标账号：`3975859721`
- 目标昵称：`靶号_01`
- LLoneBot：`7.12.15`
- QQNT：`9.9.30-48762`
- PMHQ：`127.0.0.1:13000`
- LLoneBot OneBot 反向 WebSocket：`ws://127.0.0.1:8078`

## 公开实现来源

Lagrange.Core 暴露了 `RequestFriend` 接口。公开代码里的流程是：

1. `OidbSvcTrpcTcp.0x972_6`：搜索目标账号。
2. 等待 5 秒。
3. `OidbSvcTrpcTcp.0x7c1_1`：查询/确认目标好友设置。
4. `OidbSvcTrpcTcp.0x7c2_5`：发送好友申请。

关键源码位置：

- `Lagrange.Core/Internal/Context/Logic/Implementation/OperationLogic.cs`
- `Lagrange.Core/Internal/Service/Action/RequestFriendSearchService.cs`
- `Lagrange.Core/Internal/Service/Action/RequestFriendSettingService.cs`
- `Lagrange.Core/Internal/Service/Action/RequestFriendService.cs`
- `Lagrange.Core/Internal/Packets/Service/Oidb/Request/OidbSvcTrpcTcp0x7C2_5.cs`
- `Lagrange.Core/Utility/Sign/SignProvider.cs`

`SignProvider` 中也把 `OidbSvcTrpcTcp.0x7c2_5` 标注为 `request friend`，说明它是主动好友申请通道。

## 当前 LLoneBot 环境中的差异

Lagrange 的三段式流程不能机械照搬为“三包都必须成功”。

本机在当前 QQNT/LLoneBot 环境下，重新发送只读前置查询时：

- `0x972_6` 成功返回目标账号，响应中包含 `3975859721`、`靶号_01`、`isFriend:0` 等信息。
- `0x7c1_1` 使用 Lagrange 公开字段时返回错误：
  - `errorCode: 316`
  - `errorMsg: [oidb] one of uid/openid is invaild`
  - `qq-i18n-tip-msg: 账号转换错误：存在格式非法账号`

这说明 `0x7c1_1` 在新版 QQNT/LLoneBot 环境中可能已改成 UID/OpenID 语义，Lagrange 的 UIN 字段布局已经不完全适配。

但最终 `0x7c2_5` 使用 Lagrange 字段布局后，目标电脑端已能看到过滤通知。因此当前最小可靠实现应把 `0x972_6` 作为目标存在性确认，把 `0x7c2_5` 作为实际发送动作；`0x7c1_1` 只能作为可选研究项，不能作为成功必要条件。

## `0x7c2_5` 字段布局

Lagrange 公开实现中的 body 字段如下：

| Field | 类型 | 值 | 含义 |
| --- | --- | --- | --- |
| `1` | uint32 | self uin | 发起账号 QQ 号 |
| `2` | uint32 | target uin | 目标账号 QQ 号 |
| `3` | uint32 | `1` | 固定字段 |
| `4` | uint32 | `1` | 固定字段 |
| `5` | uint32 | `0` | 固定字段 |
| `7` | string | `""` | 备注 |
| `11` | uint32 | `1` | source id |
| `12` | uint32 | `3` | sub source id |
| `18` | string | question | 验证问题/问题答案相关字段 |
| `20` | uint32 | `0` | 好友分组 |
| `26` | string | message | 申请附言 |
| `28` | uint32 | `1` | 固定字段 |
| `29` | uint32 | `1` | 固定字段 |

OIDB base 外层需要：

- `command = 0x7c2`
- `subCommand = 5`
- `body = 上表 protobuf body`
- `reserved = 1`

本次成功进入目标“已过滤通知”的请求使用：

- `sourceId = 1`
- `subSourceId = 3`
- `addSource` 在 QQNT 事件里显示为 `QQ号查找`
- `reqType = 13`
- `reqSubType = 13`
- `extWords = 请求添加对方为好友`

## 发送流程

推荐流程：

1. 查询目标账号是否存在。
   - 通过 LLoneBot 的 `get_stranger_info`，或发送 `0x972_6`。
   - 本次目标 `3975859721` 可解析为昵称 `靶号_01`。

2. 构造并发送 `0x7c2_5`。
   - 通过 LLoneBot 的 `send_pb` action。
   - 或直接通过 PMHQ WebSocket 发送 `type: "send"`。

3. 不要把 `send_pb` 返回成功视为最终成功。
   - Lagrange 的 `RequestFriendService.Parse` 在公开实现中直接返回 `Result(0)`，没有解析真实业务错误。
   - 当前验收需要额外证据。

4. 验收。
   - 强证据：目标账号在 QQ 电脑端“已过滤的通知”或普通好友申请列表中看到请求。
   - 辅助证据：发送端 `nodeIKernelBuddyListener/onBuddyReqChange` 出现 `isInitiator: true` 的请求。
   - 接受后证据：发送端好友列表包含目标 UID/UIN。

## Python 构包示例

下面是最小 `0x7c2_5` protobuf 构包逻辑。实际发送时，把生成的 hex 作为 LLoneBot `send_pb` 的 `hex` 参数。

```python
def enc_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)

def key(field: int, wire_type: int) -> bytes:
    return enc_varint((field << 3) | wire_type)

def u(field: int, value: int) -> bytes:
    return key(field, 0) + enc_varint(value)

def bs(field: int, value: bytes) -> bytes:
    return key(field, 2) + enc_varint(len(value)) + value

def s(field: int, value: str) -> bytes:
    return bs(field, value.encode("utf-8"))

def oidb(command: int, sub_command: int, body: bytes, reserved: int | None = None) -> str:
    out = bytearray()
    out += u(1, command)
    out += u(2, sub_command)
    out += bs(4, body)
    if reserved is not None:
        out += u(12, reserved)
    return out.hex()

def build_add_friend_pb(self_uin: int, target_uin: int, message: str, question: str = "") -> str:
    body = bytearray()
    body += u(1, self_uin)
    body += u(2, target_uin)
    body += u(3, 1)
    body += u(4, 1)
    body += u(5, 0)
    body += s(7, "")
    body += u(11, 1)
    body += u(12, 3)
    body += s(18, question)
    body += u(20, 0)
    body += s(26, message)
    body += u(28, 1)
    body += u(29, 1)
    return oidb(0x7C2, 5, bytes(body), reserved=1)
```

LLoneBot action：

```json
{
  "action": "send_pb",
  "params": {
    "cmd": "OidbSvcTrpcTcp.0x7c2_5",
    "hex": "<build_add_friend_pb 输出>"
  },
  "echo": "add-friend-<unique-id>"
}
```

## 本次验证证据

实际发送时间：

- `2026-05-29 22:40:55`

LLoneBot 日志显示收到：

- `send_pb OidbSvcTrpcTcp.0x972_6`
- `send_pb OidbSvcTrpcTcp.0x7c1_1`
- `send_pb OidbSvcTrpcTcp.0x7c2_5`

发送后，PMHQ/QQNT 发送端事件出现：

```json
{
  "sub_type": "onBuddyReqChange",
  "data": {
    "unreadNums": 0,
    "buddyReqs": [
      {
        "isDecide": false,
        "isInitiator": true,
        "friendUid": "u_AOpasPL9ipOnvtPlsMkoAQ",
        "reqType": 13,
        "reqSubType": 13,
        "reqTime": "1780065656",
        "extWords": "请求添加对方为好友",
        "friendNick": "靶号_01",
        "sourceId": 1,
        "isAgreed": false,
        "relation": 0,
        "addSource": "QQ号查找"
      }
    ]
  }
}
```

目标账号 QQ 电脑端“已过滤的通知”显示：

- 时间：`2026-05-29 22:40`
- 来源：`Iccc QQ号查找`
- 文案：`对方加我为好友，请谨慎同意`

这三项共同证明：申请已进入 QQ 好友申请流，但被目标端过滤。

## 电脑端和手机端是否同一路线

从协议层看，是同一路线。

原因：

- 发送动作走的是 QQ 服务端 OIDB 好友申请协议 `0x7c2_5`，不是本地 UI 自动化。
- 目标电脑端能看到该申请，说明服务端已为目标账号生成好友申请记录。
- 手机端没有普通通知，是目标客户端对过滤类通知的展示策略问题，不代表另有一条“手机专用好友申请协议”没有调用。

仍需保留的谨慎点：

- 如果用手机 QQ 作为“发起端”手动添加好友，腾讯可能根据设备、客户端类型、登录环境、历史行为给出不同风控评分。
- 但这属于风控判定差异，不是接收端有另一套好友申请路线。

因此，不需要寻找“手机端实现”来解决当前问题；当前问题是过滤/风控，不是协议未送达。

## 是否有防止被过滤的方法

截至本次验证，未发现 Lagrange 公开实现提供防过滤方法。

公开代码中可控输入只有：

- `targetUin`
- `question`
- `message`

关键风险/来源字段在 Lagrange 中是固定值：

- `SourceId = 1`
- `SubSourceId = 3`
- `Field28 = 1`
- `Field29 = 1`

这意味着 Lagrange 公开实现没有暴露类似 `securityVerify`、`ticket`、`randStr`、`token`、`bytesPermission` 或“强制普通通知”的字段。

本机 QQNT 内部 AddBuddyService 只读探测结果：

- `requestInfoByAccount` 可解析目标。
- `queryUinSafetyFlag` 以 UIN 查询返回 `status: 0`。
- `getAddBuddyRequestTag` 返回 `rsp: null`。
- `getBuddySetting`、`getSmartInfo` 直接调用仍报 `TypeError: Cannot convert undefined or null to object`，疑似依赖 UI 上下文对象。

所以目前没有证据表明存在一个公开、稳定、可由 Lagrange/LLoneBot 直接设置的“防过滤参数”。

可能影响过滤的因素应视为 QQ 服务端风控输入，而不是可靠 API：

- 发起账号信誉、年龄、历史加好友频率。
- 发起账号和目标账号是否有共同群、共同好友或真实互动上下文。
- 来源字段 `SourceId/SubSourceId`。
- 是否通过官方 UI 完成搜索页/资料页流程。
- 是否触发并完成安全验证。

其中 `SourceId/SubSourceId` 可以继续研究，但不能在没有目标端验证前宣称可防过滤。

## 集成建议

如果要把能力接入本项目，应采用保守命名和返回语义：

- 工具名可以是 `request_qq_friend`。
- 返回结果应包含：
  - `submitted: true/false`
  - `target_uin`
  - `message`
  - `may_be_filtered: true`
  - `sender_pending_evidence`
  - `verification_required: "ask recipient to check normal requests and filtered notifications"`

推荐验收文案：

```text
好友申请已提交。QQ 可能把请求放入对方的“已过滤通知”，对方手机端可能不会弹出普通通知。
请让对方同时检查 QQ 电脑端/手机端的好友申请与已过滤通知。
```

推荐保护：

- 对同一目标加好友请求做冷却，避免重复触发风控。
- 发送前先检查是否已是好友。
- 发送后记录 pending 证据，但不要自动重试。
- 目标端未确认前，不把任务状态标记为“已成为好友”。

## 后续研究方向

1. 抓取官方 QQNT UI 手动添加好友时的完整 PMHQ/PB 流程，对比 `0x7c2_5` 字段。
2. 研究新版 `0x7c1_1` 的 UID/OpenID 字段布局，确认是否能恢复好友设置查询。
3. 对比不同 `SourceId/SubSourceId` 是否影响过滤，但每次必须用目标端实际可见性验证。
4. 观察同一发起账号在不同频率、不同目标、不同共同关系下的过滤结果。

