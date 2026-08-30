# Bot Linux 环境 / 沙盒实现只读审计

审计日期：2026-08-29（Asia/Shanghai）  
项目：`E:\Aic_forQ\core`  
对象：Bot 的 `computer` / Agent Linux computer，不包含主 Bot、NapCat、模型 API 本身的资源开销。

## 结论先行

当前实现不是 WSL 1、Docker Desktop、QEMU、chroot，也不是一台由项目自行管理的完整虚拟机。它的准确结构是：

```text
Windows Bot (Quart/Python)
  -> 每次 RPC 启动一个 wsl.exe stdio bridge
  -> 专用 WSL 2 发行版 AICQ-Workspace（Ubuntu 24.04，systemd）
  -> 常驻用户级 Python broker（Unix socket）
  -> daemonless rootless Podman
  -> Ubuntu 容器 aicq-workspace-default
  -> /home/agent（持久化工作区）
```

WSL 2 本身在 Windows 下使用微软管理的轻量 utility VM 和真实 Linux 内核；项目又在该 WSL 2 发行版内部套了一层 rootless Podman 容器。因此，它是“轻量虚拟机底座 + Linux 容器”的双层方案，而不是单纯 rootfs/chroot。

对普通用户的综合评级：**较重**，但不是“很重”。

- 轻的一面：当前发行版未启动时，项目专属 CPU、内存和进程数实测均为 0；不需要 Docker Desktop 或 Podman daemon；VHDX 是稀疏文件；Bot 主进程启动不会自动启动 WSL。
- 重的一面：当前磁盘已实际分配 **8.552 GiB**；依赖硬件虚拟化、WSL 2、较新的 Store WSL、systemd，首次启用 WSL 可能需要 UAC 与重启；唤醒后有 WSL VM、systemd、broker、容器 init/sleep 和 rootless 网络辅助进程。
- “8 GiB 内存、64 GiB 磁盘”都是上限而不是启动即预留的实际占用。当前最明确的重负担是**磁盘和部署门槛**，不是停止状态下的内存或 CPU。

## 审计方法与限制

本次严格按只读边界执行：

- 读取当前代码、配置中的 `workspace` 小节、WSL 注册信息、已有控制任务元数据和微软官方文档。
- 在查询前后分别确认运行中的 WSL 发行版；`AICQ-Workspace` 始终为 `Stopped`，查询没有唤醒它。
- 未启动/停止容器，未执行 Bot 文件工具，未进入 Linux 发行版，未修改配置。
- 因此，“停止态”和磁盘是本机实测；“已唤醒空闲态”及“文件任务增量”只能由进程模型和资源边界推导，不能冒充实测值。

## 一、完整技术栈

### 1. Windows 侧

- Bot 只在 full runtime 构造 `WorkspaceService(WslWorkspaceBackend())`；构造阶段不执行 health probe，也不启动 WSL（[`src/main.py`](../src/main.py#L120-L123)，[`src/workspace/config.py`](../src/workspace/config.py#L44-L57)）。
- 每次请求启动独立的 `wsl.exe --distribution AICQ-Workspace --user aicqws --exec /usr/local/bin/aicq-workspace-bridge`，通过 stdin/stdout 传一条 NDJSON；不是长连接 Windows daemon（[`src/workspace/backend.py`](../src/workspace/backend.py#L75-L111)，[`src/workspace/backend.py`](../src/workspace/backend.py#L265-L291)）。

### 2. WSL 2 appliance

- provisioning 明确执行 `wsl.exe --install Ubuntu-24.04 ... --version 2`，当前注册信息也实测为 `VERSION 2`（[`scripts/workspace/provision-workspace.ps1`](../scripts/workspace/provision-workspace.ps1#L229-L233)）。
- 发行版使用 Ubuntu 24.04，并启用 systemd；禁用 Windows 磁盘自动挂载、Windows 可执行文件 interop、Windows PATH 注入和 GPU（[`scripts/workspace/appliance/etc/wsl.conf`](../scripts/workspace/appliance/etc/wsl.conf)）。
- WSL 2 的底层是微软管理的轻量 utility VM，而非 WSL 1 的系统调用翻译层；微软官方也将 WSL 2 定义为真实 Linux 内核 + managed VM：[Comparing WSL versions](https://learn.microsoft.com/en-us/windows/wsl/compare-versions)。

### 3. WSL 内部服务与隔离

- system service `aicq-workspace-firewall.service` 在启动时一次性应用 nftables 规则，完成后以 `RemainAfterExit=yes` 保持 unit 状态，但没有常驻 firewall 脚本进程（[`aicq-workspace-firewall.service`](../scripts/workspace/appliance/etc/systemd/system/aicq-workspace-firewall.service)）。
- linger 用户 `aicqws` 的 user systemd 常驻 `broker.py`；broker 监听 `/run/aicq-workspace/broker.sock`（[`bootstrap.sh`](../scripts/workspace/appliance/bootstrap.sh#L52-L68)，[`aicq-workspace-broker.service`](../scripts/workspace/appliance/etc/systemd/user/aicq-workspace-broker.service)，[`broker.py`](../scripts/workspace/appliance/opt/aicq-workspace/broker.py#L748-L766)）。
- rootful Podman socket/service被明确 disable + mask。容器运行依赖 daemonless rootless Podman，不依赖 Docker daemon、Docker Desktop 或常驻 Podman API（[`bootstrap.sh`](../scripts/workspace/appliance/bootstrap.sh#L36-L43)）。
- rootless 存储使用 `overlay` + `fuse-overlayfs`，网络使用无端口发布的 `slirp4netns:allow_host_loopback=false`；WSL 侧 nftables 再阻断私网/回环出站与非回环入站（[`bootstrap.sh`](../scripts/workspace/appliance/bootstrap.sh#L69-L96)，[`protocol-manifest.json`](../scripts/workspace/appliance/opt/aicq-workspace/protocol-manifest.json)，[`apply-firewall.sh`](../scripts/workspace/appliance/usr/local/lib/aicq-workspace/apply-firewall.sh)）。

### 4. 容器层

- 容器镜像是固定 digest 的 Ubuntu 基础镜像，内含 Bash、编译工具链、Git、Python 完整环境、pip/venv/uv、ripgrep、sudo 等（[`Containerfile`](../scripts/workspace/appliance/opt/aicq-workspace/image/Containerfile)）。
- 容器以 `agent:agent`（uid/gid 1000）运行，默认目录 `/home/agent`，在容器内允许 passwordless `sudo`；通过 rootless user namespace 将权限限制在 appliance 内（[`provision-container.sh`](../scripts/workspace/appliance/opt/aicq-workspace/provision-container.sh#L87-L110)）。
- 容器入口最终是 `tini -- sleep infinity`，用于保持“电脑”在线；不是每个任务新建一次容器（[`init-agent-home.sh`](../scripts/workspace/appliance/opt/aicq-workspace/image/init-agent-home.sh)，[`Containerfile`](../scripts/workspace/appliance/opt/aicq-workspace/image/Containerfile#L67-L72)）。

### 5. 不是哪些方案

| 候选 | 当前实现 |
|---|---|
| WSL 1 | 否；显式安装和当前注册状态均为 WSL 2 |
| Docker Desktop | 否；即使本机另有 `docker-desktop` 发行版，本项目不调用它 |
| 原生 Docker daemon | 否 |
| Hyper-V 完整 VM | 否；使用 WSL 2 背后的 Hyper-V/Virtual Machine Platform managed VM，不管理独立 VM 配置 |
| QEMU | 否 |
| rootfs/chroot/proot | 否 |
| 纯 Podman 容器 | 否；Podman 容器运行在专用 WSL 2 appliance 内 |
| 浏览器沙盒 | 另一路径；浏览器是 Windows 原生 Chromium + 网络 gateway，不是此 Linux 容器的第二台 VM（[`scripts/workspace/README.md`](../scripts/workspace/README.md#L82-L108)） |

## 二、启动时实际会启动的进程和服务

### Bot 自身启动

只创建 Python 服务对象，不触碰 WSL。此时不会出现项目专属 `wsl.exe`、`vmmemWSL`、Linux systemd、broker 或容器进程。

### 第一次 Linux RPC / 文件工具调用

1. Windows 临时启动 `wsl.exe`。
2. WSL 2 utility VM 启动，对应 Windows 侧通常出现/活跃 `vmmemWSL`、`wslhost.exe` 等共享 WSL 组件。
3. `AICQ-Workspace` 内启动 WSL init/systemd 及 Ubuntu 基础 system services。
4. nftables firewall oneshot 执行一次。
5. linger 用户的 `systemd --user` 启动常驻 `broker.py`。
6. 当前请求的 Python `bridge.py` 连接 Unix socket；请求结束后 bridge 与 Windows `wsl.exe` 退出。
7. broker 的 `ensure_default` 检查镜像、容器、标签及资源限制；容器若停止则执行 `podman start`（[`broker.py`](../scripts/workspace/appliance/opt/aicq-workspace/broker.py#L246-L297)）。
8. 容器内长期保留 `tini` + `sleep infinity`。rootless 容器还会有 `conmon`、`slirp4netns` 等版本相关辅助进程；Podman 本身没有常驻 daemon。

### 每次文件操作额外出现

- 一个短命 Windows `wsl.exe`。
- 一个短命 Linux `bridge.py` Python 进程。
- 若干**串行**的短命 Podman CLI 检查/更新进程。
- 一个 `podman exec` 及容器内 `aicq-file-ops` Python 进程。
- `find_files` / `search` 再启动一个 `rg` 子进程（[`broker.py`](../scripts/workspace/appliance/opt/aicq-workspace/broker.py#L648-L684)，[`file-ops.py`](../scripts/workspace/appliance/opt/aicq-workspace/image/file-ops.py#L295-L387)）。

任意 shell command 则再启动 `aicq-command-runner`，fork 新进程组并 `exec /bin/bash command.sh`（[`command-runner.py`](../scripts/workspace/appliance/opt/aicq-workspace/image/command-runner.py)）。

### 一个容易忽略的唤醒入口

Web 设置页的 `GET /api/computer` 调用 `status_payload()`，后者会调用 `probe()`；已安装时，`probe()` 会执行多条 `wsl.exe --distribution ... --exec ...` 深度检查。因此，**仅打开/刷新 Agent 电脑状态页也可能唤醒 WSL 发行版**，虽然它不会主动创建镜像或容器（[`src/web/routes_workspace.py`](../src/web/routes_workspace.py#L32-L40)，[`src/workspace/control.py`](../src/workspace/control.py#L372-L477)，[`src/workspace/control.py`](../src/workspace/control.py#L639-L647)）。

## 三、空闲不跑任务时的 CPU、内存、磁盘和进程数

### A. 当前本机“未唤醒/停止态”实测

| 指标 | 本机实测 | 解释 |
|---|---:|---|
| `AICQ-Workspace` 状态 | `Stopped` | 查询前后均为 Stopped，未被本次审计唤醒 |
| 项目专属进程数 | **0** | 未发现命令行指向 `AICQ-Workspace` / `aicq-workspace` 的 WSL/Podman 进程 |
| 项目专属 CPU | **0** | 无项目进程 |
| 项目专属内存 | **0 MiB** | 无 `vmmemWSL`，无项目进程；这里不把共享 Windows 服务硬算给项目 |
| `wslservice` | WS 30.24 MiB；Private 7.13 MiB | Windows 全局 WSL 服务，不属于此项目独占 |
| `vmcompute` | WS 15.43 MiB；Private 2.85 MiB | Windows 全局虚拟化服务，不属于此项目独占 |
| 以上共享服务 5 秒 CPU 增量 | **0.000 s** | 5 秒短采样，不能代表长期上限 |
| `ext4.vhdx` 文件长度 | **9.616 GiB** | 稀疏 VHDX 当前文件长度，不是 64 GiB 预分配 |
| `ext4.vhdx` NTFS 实际分配 | **8.552 GiB** | `GetCompressedFileSizeW` 只读测得的占盘量 |
| 虚拟磁盘配置上限 | **64 GiB** | 上限；磁盘只能扩容，不能原地缩容 |

VHDX 当前路径：`E:\Aic_forQ\core\data\AICQ-Workspace\ext4.vhdx`。微软说明 WSL 2 的发行版文件存于 ext4 VHDX，VHD 会随实际数据增长：[How to manage WSL disk space](https://learn.microsoft.com/en-us/windows/wsl/disk-space)。

### B. 已唤醒但无任务的状态

**没有本次实测值。** 原因是发行版在审计开始时已停止；为测量这组数字必须主动唤醒 WSL/容器，违背本次只读、不启动服务的边界。项目也没有保存历史 RSS、CPU 或进程数遥测。

由代码能确定：

- CPU 应接近事件驱动空闲：broker 阻塞等待 Unix socket，容器主进程是 `sleep infinity`，不存在 Podman daemon 轮询；systemd/日志/网络仍会有少量周期活动。
- 内存不是固定占 8 GiB。8 GiB 是容器 cgroup 上限；WSL 2 VM 按需增长。当前用户全局 `.wslconfig` 另设 `memory=44GB`、`swap=32GB`，这只是所有 WSL 2 发行版共享 VM 的上限/交换空间，不代表 AICQ 实占。
- 为部署规划可暂按 **约 0.3–0.8 GiB 的 WSL/Ubuntu/systemd/broker/空容器量级**估算，但这不是本机实测，文件缓存、WSL 版本和其他同时运行的发行版都可能使 `vmmemWSL` 更高。
- 进程数只能给拓扑下限，不能给当前精确值：至少包含 systemd/WSL init、user systemd、broker、`tini`、`sleep` 及容器 monitor/network helper，再加 Ubuntu 基础服务。规划上应按“十几个到几十个 Linux/辅助进程”看待，而不是 2–3 个；需实际唤醒后用 `ps`/`systemd-cgtop` 才能定量。

微软官方当前默认 `.wslconfig` 的 WSL 2 VM 内存上限为宿主内存 50%，并支持 `autoMemoryReclaim`；本机的 44 GiB 是用户覆盖值：[Advanced settings configuration in WSL](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)。

## 四、Bot 执行文件处理时的资源增量

### 结构化文件工具：`read_file` / `write_file` / `find_files` / `search`

| 资源 | 代码可推导的增量 | 可信度 |
|---|---|---|
| 进程 | 峰值通常临时增加约 **4–6 个跨 Windows/Linux/容器进程**；search 再加 `rg` | 中；具体 helper 随 WSL/Podman 版本变化 |
| 内存 | 小文件操作可用 **+30–150 MiB 短时工作集**作规划量级；大型目录搜索产生的 Linux page cache 可能明显更高 | 低到中；未实测，不能写成保证值 |
| CPU | 简单读写是短促低负载；`rg` 搜索可短时使用多个核，但被容器 **4 CPU** 上限约束 | 高（边界），低（实际百分比） |
| 持久磁盘 | read/find/search 通常不增加持久数据；write 增加实际文件大小，原子替换会短时保留临时副本 | 高 |
| 输出/输入边界 | 文本写入最大 1 MiB；命令 stdin 最大 1 MiB；命令输出 spool 最大 64 MiB | 高（代码硬限制） |

文件发送/导出是另一条路径：它把 `/home/agent` 中的一个文件流式复制到 Windows 临时目录，临时磁盘增量约等于被导出文件大小，context 结束后删除（[`src/workspace/service.py`](../src/workspace/service.py#L200-L250)，[`src/workspace/backend.py`](../src/workspace/backend.py#L130-L241)）。

### 任意 `computer.command`

不能用“小文件处理增量”代表任意命令。Bot 可以在容器中编译、pip/apt 安装、运行服务器或批量生成文件。硬边界是：

- 4 CPU；
- 8 GiB 容器内存；
- 1024 PIDs；
- 64 GiB 虚拟磁盘上限（其中已使用/占盘需另测）；
- 单命令捕获输出最多 64 MiB。

因此，普通读写通常只增加几十 MiB 和短促 CPU；构建、安装或大规模搜索则可以接近完整资源上限。没有任务类型、数据规模和缓存状态时，无法给一个负责任的统一增量数字。

## 五、宿主系统硬性要求与权限

### 项目代码的实际要求

- **必须是 WSL 2**；需要 BIOS/UEFI 硬件虚拟化，Windows 的 Virtual Machine Platform 可用。
- 不要求完整 Hyper-V 管理角色，也不要求 Windows Pro；Windows Home 只要能运行 WSL 2 即可。
- 通用 WSL 安装命令的微软最低线是 Windows 10 2004 / Build 19041 或 Windows 11：[Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)。
- 本项目的实际门槛更高：脚本要求 `wsl --help` 同时支持 `--location`、`--name`、`--vhd-size`、`--manage`、`--set-sparse`。微软文档说明 `wsl --manage` 需要 **WSL 2.5+**，因此仅有老版 inbox WSL 不够（[`provision-workspace.ps1`](../scripts/workspace/provision-workspace.ps1#L396-L419)，[WSL disk management](https://learn.microsoft.com/en-us/windows/wsl/disk-space#expand-vhd-size-using-wsl---manage)）。
- systemd 需要 WSL 0.67.6+；当前 `--manage` 门槛已高于它：[Use systemd with WSL](https://learn.microsoft.com/en-us/windows/wsl/systemd)。
- 当前清单只显式记录了 amd64 platform digest，代码中未见 ARM64 专门验证分支；因此本次只能确认 **x64/amd64 Windows** 已被当前实现和本机状态覆盖，不能把 ARM64 列为已验证支持。
- 首次 provisioning 前检查至少 20 GiB 宿主空闲空间，并要求本机固定磁盘路径，拒绝 UNC、可移动盘、盘符根目录和受保护目录（[`provision-workspace.ps1`](../scripts/workspace/provision-workspace.ps1#L421-L428)，[`src/workspace/config.py`](../src/workspace/config.py#L94-L151)）。

### 管理员权限

- 如果 WSL/Virtual Machine Platform 尚未就绪，脚本会用 UAC 提升执行 `wsl --install --no-distribution`，且可能要求重启；微软的 WSL 安装步骤也要求首次启用特性时使用管理员 PowerShell（[`provision-workspace.ps1`](../scripts/workspace/provision-workspace.ps1#L396-L414)，[Microsoft WSL install](https://learn.microsoft.com/en-us/windows/wsl/install)）。
- WSL 已就绪后，发行版注册、日常 Bot RPC、rootless Podman 和普通维护以当前 Windows 用户运行，不要求 Bot 常驻管理员权限。
- 本次审计会话本身是非管理员；无法直接查询 optional feature 状态，但当前 WSL 2 发行版可注册且 `HypervisorPresent=True`，已足以证明本机运行前提存在。

### 当前宿主实测

- Windows 11 专业版，Build 26200，x64。
- WSL 2.7.1.0，Linux kernel 6.6.114.1；`AICQ-Workspace` 注册为 WSL 2。
- CPU：Ryzen 7 9800X3D，8 核/8 逻辑处理器；固件虚拟化开启；Windows 报告 hypervisor 已存在。
- 宿主可见内存约 47.56 GiB。

## 六、常驻还是按需启动；首次启动与唤醒耗时

### 生命周期判断

准确描述是：**Bot 侧按需启动，但首次唤醒后倾向常驻。**

- Bot 进程启动：完全惰性，不启动 WSL。
- 第一次 computer RPC：启动 WSL；broker 可启动已存在但停止的容器，不会由模型路径构建镜像或创建容器。
- 一旦 WSL 启动：linger user service 的 broker 和容器中的 `sleep infinity` 会保留进程，因此不能依赖 WSL 的“无进程空闲自动关机”来及时回收；代码没有普通空闲超时停止逻辑。
- 手工“Restart”会 `wsl --terminate AICQ-Workspace`，随后又立即启动并验证，所以它不是“停止并保持停止”（[`workspace-maintenance.ps1`](../scripts/workspace/workspace-maintenance.ps1#L29-L38)）。
- 设置页状态 probe 也可能提前唤醒发行版。

### 本项目已有历史耗时

以下是 `data/workspace-control/jobs/*.json` 中已有成功任务的墙钟时间；它们是历史记录，不是本次执行：

| 动作 | 成功样本 | 墙钟时间 | 可代表什么 |
|---|---:|---:|---|
| 完整 build | 2 | **90 秒、2894 秒（48 分 14 秒）** | 强烈受 Ubuntu/镜像下载、APT/pip、网络和缓存影响；不能只报最快值 |
| restart + 全套 verify | 1 | **11.54 秒** | 旧历史环境的一次完整重启验证；可作“完整 ready”参考，不等于纯 WSL boot |
| 当前协议 upgrade + verify | 1 | **38.78 秒** | 2026-07-18 当前安装路径的一次升级 |
| apply resources + verify | 多个 | 最近成功样本 **14.66 秒** | 资源更新与验证，不是普通唤醒 |

项目没有单独记录“冷 WSL boot”“容器 start”“首个 read_file”三个阶段的耗时，因此无法给出本机实测冷唤醒值。微软对普通 WSL 发行版的说明是首次解压后，后续 launch 通常少于一秒；本项目还要等待 systemd、broker、Podman 检查/容器启动，首个 Bot 工具的完整 ready 时间会更长：[Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)。在做真实基准前，只能把“数秒量级”视为估计，把 11.54 秒历史 restart+verify 视为一个带完整验证的参考值。

## 七、文件系统与 workspace 实际位置

### Windows 物理位置

当前配置：

```yaml
workspace:
  enabled: true
  install_root: E:\Aic_forQ\core\data
  resources:
    cpus: 4
    memory_gib: 8
    disk_gib: 64
```

发行版安装目录实际为：

```text
E:\Aic_forQ\core\data\AICQ-Workspace\
└── ext4.vhdx
```

### Linux 内部位置

```text
WSL appliance:
  /var/lib/aicq-workspace/home      # 持久数据
  /var/lib/aicq-workspace/commands  # 命令元数据/输出 spool

Podman container:
  /var/lib/aicq-workspace/home -> bind mount -> /home/agent
  /var/lib/aicq-workspace/commands -> /run/aicq-workspace/commands
```

模型看到的 workspace 就是 `/home/agent`。它不是 Windows 项目目录，也不是 `E:\Aic_forQ\core` 的 bind mount；Windows 盘 automount 和 interop 均被关闭，provisioning 资产通过 tar/stdin 传入，仓库不会挂进 appliance（[`scripts/workspace/README.md`](../scripts/workspace/README.md#L8-L27)，[`provision-container.sh`](../scripts/workspace/appliance/opt/aicq-workspace/provision-container.sh#L101-L109)，[`provision-workspace.ps1`](../scripts/workspace/provision-workspace.ps1#L328-L347)）。

容器系统层和 Podman OCI layers 位于同一个 `ext4.vhdx` 中。`/home/agent` 在更新/重建系统容器时保留；容器外、`/home/agent` 之外通过 `sudo` 安装的内容不保证跨系统重建保留。

文件发送时，单个文件才会临时导出到 Windows `aicq-workspace-send-*\payload.bin`，发送完成后删除；这不是常驻 workspace 的第二份镜像。

## 八、当前方案“重”的真正来源

按当前可验证证据排序：

1. **磁盘是最明确的重项。** 当前 VHDX 已实际分配 8.552 GiB。来源至少包括一套 Ubuntu WSL appliance、另一套 Ubuntu 容器 rootfs、完整编译/Python 工具链、Podman layers/build cache 和 Agent 持久文件。因本次不唤醒发行版，无法用 `podman system df` / `du` 把 8.552 GiB 精确拆分。
2. **部署门槛较重。** 需要硬件虚拟化、WSL 2.5+、systemd、固定磁盘、首次 UAC/可能重启，并需要下载 Ubuntu distro、Ubuntu OCI image 和两轮工具包。
3. **唤醒后的 WSL VM 与常驻服务是内存来源。** 不是 Podman daemon；主要是 WSL 2 VM + Ubuntu/systemd + broker + 空容器 + rootless network/monitor，再叠加 Linux page cache。
4. **当前 44 GiB WSL global memory cap 容易造成“可能很重”的观感，但不是实际占用。** 容器自身仍限 8 GiB；44 GiB 是所有 WSL 2 发行版共享 VM 的上限。
5. **任务本身可能远重于框架。** 任意 command 能进行编译、安装和长期服务；这时负载来自任务，而不是 bridge/broker。

不是主要来源的项目：

- 停止状态的 CPU/内存：实测为 0 项目进程。
- Docker Desktop：不在此调用链中。
- Podman daemon：已禁用。
- 64 GiB 磁盘和 8 GiB RAM 配额：都不是预分配常驻占用。

## 九、在不大幅牺牲功能前提下的优化空间

### 优先级 A：高收益、功能损失小

1. **增加空闲 lease/自动休眠。** 无运行 command、无导出、无 control job 并持续空闲一段时间后，停止容器并 `wsl --terminate AICQ-Workspace`。文件和容器 writable layer 都保留；代价是下次工具多一次冷启动。必须和后台 command 生命周期绑定，不能按固定时间粗暴终止。
2. **让状态页浅探测不唤醒 WSL。** 默认只读注册表、managed marker、`wsl --list --running`；仅当已经运行或用户点“深度检查”时执行 distro 内 probe。当前实现打开设置页就可能唤醒，这是不必要的后台驻留入口。
3. **将 broker 改为无任务时退出或 socket/on-demand 模式。** 保留有活动 command 时的常驻监控，无任务后退出，让 WSL 自身进入可回收状态。此改动比简单定时 terminate 更干净，但需要重新处理并发 RPC 和 command 恢复。
4. **增加资源/生命周期可观测性。** 在设置页记录 cold boot、container start、首个 RPC、idle RSS、进程数和 VHD allocated bytes。当前无法精确回答运行态问题，本质上是没有遥测，而非指标不可测。

### 优先级 B：可配置优化

5. **提供“轻量/标准”资源预设。** 普通文本文件工具可从 2 CPU / 4 GiB / 32 GiB 起步；编译或重任务再切 4 CPU / 8 GiB / 64 GiB。当前 UI/校验本来就允许最低 2 GiB RAM、32 GiB disk。CPU/内存可原地更新，磁盘缩容仍需重建。
6. **谨慎降低全局 `.wslconfig`。** 当前 `memory=44GB`、`swap=32GB` 对 47.56 GiB 宿主非常宽松。若其他 WSL/Docker 工作负载允许，可考虑 12–16 GiB memory、4–8 GiB swap，并使用当前 WSL 的自动内存回收。注意 `.wslconfig` 影响所有 WSL 2 发行版，不应由本项目静默修改。
7. **显式清理过期 Podman build cache/旧镜像并压缩 VHDX。** 只保留当前容器和当前 digest，先用 `podman system df` 证明可回收对象，再做受控清理和 VHD compaction。降低 64 GiB 上限本身不会回收当前 8.552 GiB；VHD shrink 也不是现有原地操作。

### 不建议作为“轻量化”的替代

- 改为 Docker Desktop：仍需要 WSL/VM，还会引入 Desktop 常驻组件，通常不会降低部署门槛。
- 改为完整 Hyper-V/QEMU VM：更重。
- 改为 WSL 1：rootless Podman、完整 syscall/cgroup/systemd 兼容不满足。
- 直接去掉容器只留 WSL rootfs：会减少一层磁盘和进程，但显著削弱“Agent 有 sudo、宿主 appliance 仍受保护”的隔离边界，不属于“不大幅牺牲功能/安全”的优化。

## 十、最终等级与判断依据

### 评级：较重

| 维度 | 等级 | 判断依据 |
|---|---|---|
| 停止态 CPU/内存 | 很轻 | 项目进程 0、项目内存 0、项目 CPU 0 |
| 唤醒后空闲 | 中等（待实测） | WSL VM + systemd + broker + rootless 容器常驻；无 Docker/Podman daemon |
| 普通文件任务增量 | 较轻到中等 | 短命 bridge/Podman/file-op/rg；小文件受限，但 search 会推高 cache/I/O |
| 任意 command 上限 | 较重 | 4 CPU、8 GiB、1024 PIDs，可运行编译/安装/服务 |
| 磁盘 | 较重 | 当前实际 8.552 GiB，且是双层 Ubuntu/OCI/持久 home 合计 |
| 部署门槛 | 较重 | 硬件虚拟化、WSL 2.5+、systemd、UAC/可能重启、首次网络构建 |
| 隔离能力 | 较强 | WSL 2 + rootless Podman + userns + nftables + no mounts/interop/GPU/ports |

如果只看“电脑未使用时”，它是很轻；如果看普通用户从安装、占盘、虚拟化前提到唤醒后常驻的完整体验，它应归为**较重**。之所以不评“很重”，是因为它不是传统完整 VM、停止时没有项目专属内存/CPU、Podman daemon 不常驻，而且 WSL 2 与 VHDX 都是按需/稀疏机制。

## 若以后允许做一次受控实测，建议的基准

为了补齐本次无法取得的三组数字，应在用户明确允许启动后按固定顺序测量，并在最后恢复停止状态：

1. 停止态：记录进程、`vmmemWSL`、VHD allocated bytes。
2. cold `health/ensure_default`：分段记录 WSL boot、broker ready、container start、首 RPC。
3. 已唤醒空闲 60 秒：记录 CPU、RSS/private、Linux `ps` 数、cgroup memory。
4. 固定小文件：1 MiB write、read、1000 文件 `find_files`、固定词 `search`，分别记录峰值和完成时间。
5. 运行后空闲 5 分钟：检查 page cache 回收与是否仍常驻。
6. 显式停止容器并 terminate distro，确认进程与内存回到基线。

只有这组受控数据才能把本文中的 `0.3–0.8 GiB`、`+30–150 MiB` 和进程数规划区间替换成本机可靠实测。
