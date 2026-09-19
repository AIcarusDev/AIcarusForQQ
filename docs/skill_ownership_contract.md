# Skill 归属契约

每个 namespace 通过 `skill` 字段绑定一个 skill。`src/skills/registry.py` 中的
`SKILL_KINDS` 声明该 skill 的归属：`user` 或 `project`。新增 skill 时，应先在此处
声明类型，再建立对应文件并绑定 namespace。漏声明的 skill 在运行时记录警告，
按 `project` 处理；它不能获得用户副本，也不能通过用户正文接口保存。

## user skill

目前只有 `qq-social-style`。仓库追踪 `SKILL.md.template` 作为首次安装的初始正文，
运行时在同目录生成被 Git 忽略的 `SKILL.md`。已有用户正文始终优先，模板更新不会
覆盖它。Web 设置页只读写用户正文；保存后清除正文缓存，使下一轮加载使用新内容。
用户可在本地添加 references；社交 skill 的 references 目录默认被 Git 忽略。

## project skill

`core-chat`、`project-source`、`computer`、`video`、`qq-file` 使用仓库追踪的
`SKILL.md` 作为唯一正文。加载器不从模板生成副本，也不以模板作后备。修改项目
skill 时只编辑该文件；需要随版本分发的 references 也应纳入 Git。运行进程会缓存
skill 正文，部署项目 skill 的更改后须重启进程。

skill 的归属只决定文件来源和用户保存权限，不改变 namespace 生命周期、
`<skills>` 渲染格式或按需读取 reference 的方式。本契约不提供 Agent 自主修改
skill 的工具。
