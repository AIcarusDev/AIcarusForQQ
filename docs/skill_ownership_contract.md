# Skill 归属契约

每个 namespace 通过 `skills` 列表按顺序绑定多个 skill，兼容原有 `skill` 单值字段。
两者同时声明时先取 `skill`，再取 `skills`，按 ID 去重；不同 namespace 共享同一
skill 时，正文只渲染一次，只要还有绑定它的 namespace 激活，它就继续可见。
正文渲染、namespace 管理回执和 reference 读取权限使用相同的绑定列表。
`src/skills/registry.py` 中的
`SKILL_KINDS` 声明该 skill 的归属：`user` 或 `project`。新增 skill 时，应先在此处
声明类型，再建立对应文件并绑定 namespace。漏声明的 skill 在运行时记录警告，
按 `project` 处理；它不能获得用户副本，也不能通过用户正文接口保存。

## user skill

目前只有 `qq-social-style`。仓库追踪 `SKILL.md.template` 作为首次安装的初始正文，
运行时在同目录生成被 Git 忽略的 `SKILL.md`。已有用户正文始终优先，模板更新不会
覆盖它。Web 设置页只读写用户正文；保存后清除正文缓存，使下一轮加载使用新内容。
用户可在本地添加 references；社交 skill 的 references 目录默认被 Git 忽略。

## project skill

`qq-social-tools`、`core-chat`、`project-source`、`computer`、`video`、`qq-file` 使用仓库追踪的
`SKILL.md` 作为唯一正文。加载器不从模板生成副本，也不以模板作后备。修改项目
skill 时只编辑该文件；需要随版本分发的 references 也应纳入 Git。运行进程会缓存
skill 正文，部署项目 skill 的更改后须重启进程。

skill 的归属只决定文件来源和用户保存权限，不改变 namespace 生命周期、
`<skills>` 渲染格式或按需读取 reference 的方式。本契约不提供 Agent 自主修改
skill 的工具。

## QQ 社交拆分

`qq_social` 按顺序挂载 `qq-social-tools`（project）和 `qq-social-style`（user）。
前者只记录操作方式、平台事实和工具组合方法，不规定何时发言、社交姿态或表达偏好；
后者负责语气、节奏、参与话题和氛围判断，不承担工具操作手册。
工具声明保留用途、参数及关键限制；较长的操作流程和示例写入项目 skill。
跨 namespace 可用的工具仍需保留其独立调用所需说明。

本次同时清理仓库模板和开发机已有用户正文中的工具操作说明。其他安装已有的
`SKILL.md` 不会被模板更新覆盖：备份后可自行移除操作说明，或删除旧用户文件，
由新模板重新生成，再恢复个人表达偏好。无需改动用户 references。
