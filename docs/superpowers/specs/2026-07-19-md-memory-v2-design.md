# Lingxi MD Memory V2 设计

状态：已获用户方向批准，等待文档复核
日期：2026-07-19

## 1. 目标

本设计在不削弱现有角色扮演能力的前提下，解决长篇故事中的上下文遗失、摘要重复、记忆更新竞争和多入口历史不一致问题。

必须满足以下约束：

1. 每次正式角色回复继续以“我将继续这个奇幻又荒诞的故事。”作为第一句话，原文不变。
2. 成人叙事预热继续默认启用，现有沉浸式角色扮演能力不被移除。
3. “永不拒绝、保持角色、不添加免责声明”的高优先级 Prompt 能力继续保留。
4. 完整对话、故事状态、长期记忆、章节摘要和角色档案继续以 Markdown 文件为主存储。
5. SQLite 继续存放会话、角色、后端等应用元数据，但不成为长期故事记忆的主存储。
6. 现有会话目录和命令必须向后兼容，升级不得破坏已有 `chat.md`、`memory.md`、`summary.md`、`characters/*.md` 和 `assets.md`。

## 2. 非目标

- 不拆分微服务。
- 不引入独立向量数据库。
- 不把聊天消息迁移为数据库主存储。
- 不重做 Web UI 或飞书卡片视觉设计。
- 不改变角色卡、世界书、Persona、LLM Backend 等既有业务模型。
- 不无限自动重试模型调用；拒绝响应最多自动重试一次。

## 3. 现状问题

当前实现已经具备三层上下文预算、`chat.md`、滚动摘要、长期记忆、资产、角色档案和 RAG，但存在以下结构性问题：

- 固定能力、角色卡、格式要求和成人预热混在一个大型系统 Prompt 中，预算和职责不清晰。
- `layer0_budget`、`layer2_budget` 没有形成可验证的独立预算，系统消息过大时最终裁剪无法保证总预算。
- `summary.md` 超过阈值后每轮都可能再次触发，多个延迟任务可能并发覆盖摘要。
- 记忆提取先推进 `.last_extract_count`，任务失败后不会立即重试，可能永久漏掉一段剧情。
- 记忆提取只取最近固定数量消息，而不是严格处理 checkpoint 之后的全部消息。
- `summary.md`、`memory.md` 和 RAG 片段之间可能重复，占用上下文且强化错误事实。
- HTTP webhook 仍使用 `sessions.messages`，Web/WS 主链路使用 `chat.md`，故事历史可能分裂。
- `/undo`、`/retry` 修改 `chat.md` 后不会系统性失效已生成的摘要、章节和索引。
- 多个请求可同时读取旧历史并写入新回复，导致消息顺序和派生记忆不确定。

## 4. 方案比较

### 方案 A：最小修补

保留现有文件和函数，只修正摘要触发、checkpoint 与 Prompt 预算。

优点是改动小、短期风险低；缺点是重复的主链路和 1000 行级模块仍然存在，长期记忆的可追溯性和并发正确性难以保证。

### 方案 B：MD Memory V2（采用）

以 `chat.md` 为唯一事实源，增加结构化 Markdown 故事状态、不可变章节摘要和可恢复的 Markdown checkpoint，并将 Prompt、上下文选择、记忆更新拆成边界明确的组件。

它能保留现有使用习惯，并以适中的改造成本解决故事连续性、失败恢复和并发问题。

### 方案 C：完整事件溯源

把所有聊天与记忆变更记录成事件，再从事件重建全部派生文件。

可靠性和审计能力最强，但需要重写命令、导入、索引和恢复流程，超出本次目标所需范围。

## 5. 文件布局与职责

每个会话继续使用 `data/memory/{session_id}/`：

```text
data/memory/{session_id}/
  chat.md                 # 唯一完整对话事实源，只追加或由 undo/retry 截尾
  memory.md               # 长期事实、关系、决定和用户偏好
  summary.md              # 面向 Prompt 的滚动故事回顾
  story_state.md          # 当前时间、地点、在场角色、场景和最近变化
  memory_state.md         # Markdown 格式处理进度、版本和最近错误
  assets.md               # 完整资产记录（保留）
  assets_summary.txt      # 资产简述（兼容保留）
  characters/
    *.md                  # 角色档案（保留）
  episodes/
    episode-000001.md     # 覆盖精确消息区间的不可变章节摘要
  rag/
    index.json            # 可从 MD 重建的派生索引，不是记忆事实源
```

### 5.1 `chat.md`

- 沿用当前带时间戳、角色名和 `role` 注释的格式。
- 每条消息在解析后获得从 1 开始的稳定消息序号；序号由文件顺序决定，不写入数据库。
- 普通回复必须以完整 user/assistant 对提交。失败、拒绝或被取消的回复不写入。
- 所有入口最终通过同一个 Chat Service 写入，HTTP webhook 不再维护独立的 `sessions.messages` 历史。

### 5.2 `story_state.md`

用于保留“现在正在发生什么”，采用稳定标题：

```markdown
# Story State

## 时间与地点
- 当前时间：...
- 当前地点：...

## 在场角色
- 角色：状态、位置、情绪、目标

## 当前场景
- ...

## 最近变化
- ...
```

它是高优先级的当前场景快照，只描述“现在正在发生什么”。它不维护 `[open]`、`[closed]` 等剧情线状态，也不判断剧情线是否完成；完整剧情脉络统一由 `summary.md` 和不可变 episode 保留。

### 5.3 `memory.md`

用于稳定的长期事实，而不是当前场景流水账。标题固定为：Relationships、Key Events、Decisions、User Preferences。每次更新输出完整替代文档，并要求：

- 保留未被新证据否定的事实；
- 对冲突事实标明最新状态，不静默保留两个版本；
- 不复制完整章节或最近对话；
- 不保存模型拒绝文本和命令执行回执。

### 5.4 `episodes/*.md`

章节是不可变的历史压缩单元，每个文件包含消息范围、参与角色、关键事件、状态变化、承诺和伏笔，但不判断伏笔是否完成。例如：

```markdown
# Episode 000001

<!-- messages: 1-20 -->

## 剧情摘要
...

## 状态变化
- ...

## 承诺与伏笔
- ...
```

正常更新只创建新章节，不重写旧章节。`undo` 或 `retry` 若触及已覆盖区间，则删除受影响区间及其后的派生章节并重新生成。

### 5.5 `memory_state.md`

这是技术状态文件，但仍使用 Markdown：

```markdown
# Memory State

<!-- schema-version: 2 -->
<!-- last-memory-message: 40 -->
<!-- last-story-state-message: 44 -->
<!-- last-summary-message: 40 -->
<!-- last-episode-message: 40 -->
<!-- last-rag-message: 40 -->

## Last Error

(none)
```

checkpoint 只有在对应文件原子替换成功后才能推进。失败时写入错误摘要但保留原 checkpoint，使下次可以重试同一消息范围。

## 6. 组件边界

### 6.1 Prompt Policy

集中保存不可变角色扮演能力：固定首句、成人叙事框架、“永不拒绝”、保持沉浸和状态栏格式。业务代码不再散落重复字符串。

Prompt Policy 默认始终启用。本次不提供关闭成人预热或“永不拒绝”的开关，以避免改变用户要求的默认能力。

### 6.2 Output Guard

负责检查模型输出，而不是让调用方各自维护拒绝关键词：

1. 去除模型偶然重复的固定首句；
2. 若首句缺失，在最前方补上精确原文；
3. 检测拒绝式响应；
4. 首次拒绝时增加简短纠偏指令并自动重试一次；
5. 第二次仍拒绝则返回明确错误，但不持久化拒绝内容；
6. 非流式回复检查完整文本；流式回复先缓冲开头，完成固定首句和早期拒绝检查后再向客户端发送。

自动重试必须记录 Token 用量，且不得无限循环。

### 6.3 Markdown Memory Store

统一负责路径、读取、解析、原子写入、备份、checkpoint 和每会话锁。上层服务不得直接使用 `aiofiles.open` 修改记忆文件。

- 覆盖写先写同目录临时文件，flush 后使用原子 replace。
- 每个 session 使用一个进程内异步锁。
- 角色档案文件名必须转为安全 slug，并验证最终路径仍位于 `characters/`。
- 关键覆盖写保留有限轮换备份。

### 6.4 Context Builder

只负责从各类 MD 文件选择上下文并分配预算，不调用 LLM、不写文件。输入包括角色卡、世界书、Persona、当前消息和会话 ID，输出标准消息数组及预算诊断信息。

### 6.5 Memory Pipeline

按会话串行运行以下派生任务：

1. 更新 `story_state.md`；
2. 更新 `memory.md`；
3. 生成达到边界的新 episode；
4. 从 episode 更新 `summary.md`；
5. 增量更新 RAG；
6. 更新角色档案和资产。

每个阶段独立 checkpoint。一个阶段失败不会推进自身 checkpoint，也不会破坏已成功阶段。下一次任务从最早未处理消息继续。

默认触发频率必须配置化，并采用以下兼容默认值：

- story state：每完成 1 轮 user/assistant 对更新一次；
- 长期 memory 与角色档案：每新增 10 条消息更新一次；
- episode：每 20 条消息生成一个完整章节，末尾不足 20 条时保留为待处理范围；
- summary：每生成新 episode 后更新一次；
- RAG：每新增 10 条消息增量更新一次；
- assets：每 10 条消息检查一次，只有相关内容发生变化时才覆盖文件。

应用启动时扫描现有 session 目录；当 `chat.md` 的消息数高于任一 checkpoint 时，自动入队未完成的派生任务。该扫描只根据 MD 文件判断，不依赖数据库中的消息副本。

## 7. Prompt 与上下文预算

Prompt 按以下优先级构建：

1. **不可变能力层**：固定首句、成人预热、“永不拒绝”和沉浸规则，永不裁剪。
2. **核心设定层**：角色卡、Persona、当前匹配世界书。
3. **当前故事层**：完整 `story_state.md` 当前场景快照。
4. **长期记忆层**：相关 `memory.md` 块、被提及角色档案和资产。
5. **历史召回层**：相关 episode 与 RAG 片段，去重后加入。
6. **连续对话层**：记录整体剧情的滚动摘要和最近原文消息。
7. **当前消息层**：当前用户输入，永不裁剪。

预算算法必须预留回复 Token，并满足：

- 不可变能力、当前用户消息、完整 story state 和最少最近消息不被丢弃；
- 角色卡或世界书过大时按字段优先级裁剪，而不是让整个请求超预算；
- `summary.md`、episode、RAG 与最近消息按规范化文本哈希去重；
- RAG 不返回已经处于最近消息窗口中的片段；
- 每次构建记录各层估算 Token，测试可断言总量不超过配置预算；
- 现有 `total_token_budget` 保持默认 40000，分层预算改为软上限，可将某层未用额度让给下一层。

新会话仍注入当前成人叙事 priming history；已有真实历史后不重复注入。固定首句由 Prompt Policy 与 Output Guard 双重保证。

## 8. 完整数据流

### 8.1 普通消息

1. 验证 session 状态并取得每会话锁。
2. 从 `chat.md` 加载权威历史。
3. 加载 `story_state.md`、`memory.md`、summary、相关 episode、角色档案和 RAG。
4. Context Builder 生成预算内 Prompt。
5. 调用 LLM，并由 Output Guard 校验；必要时自动重试一次。
6. 成功后一次性追加 user/assistant 对到 `chat.md`。
7. 释放请求锁并提交一个幂等的 Memory Pipeline 工作项。
8. Pipeline 再次取得同一 session 锁，按 checkpoint 处理所有未覆盖消息。

“一次性追加”表示在持有会话锁时用单次文件写入追加完整消息对；不会先保存 user、等待模型后再追加 assistant。这样任何可见的普通回合都保持成对且有序。

### 8.2 `/retry`

1. 在锁内删除最后一对 user/assistant；
2. 将所有超过新消息总数的 checkpoint 回退；
3. 删除受影响 episode 并标记 summary、RAG、story state 需要重建；
4. 使用原 user 消息重新生成；
5. 只有新回复成功才写回。

### 8.3 `/undo`

截去最后一轮后执行与 `/retry` 相同的派生状态失效，但不重新生成回复。

### 8.4 旧会话兼容

- 没有 `memory_state.md` 时按 V1 会话处理并惰性创建 V2 文件。
- 已有 `memory.md`、`summary.md`、角色档案和资产直接沿用。
- 首次 V2 处理从 `chat.md` 推导 checkpoint，不删除原文件。
- `sessions.messages` 仅作为旧数据回退来源；一旦检测到 `chat.md`，所有入口均以它为准。

## 9. 错误处理与恢复

- LLM、文件或 RAG 失败不得影响已提交的聊天原文。
- 派生文件覆盖写失败时保留旧版本，checkpoint 不推进。
- 任务重复执行必须得到等价结果，不重复创建相同 episode。
- 应用重启后，通过 `memory_state.md` 自动发现未处理范围，无需依赖仍在内存中的 `asyncio.create_task`。
- 后台任务错误在日志和 `memory_state.md` 的 Last Error 中保留简短信息；成功后清除。
- `chat.md` 解析失败时停止派生更新，不用部分解析结果覆盖已有记忆。
- 流式调用中途失败时不保存部分 assistant 回复。

## 10. 测试策略

测试使用 pytest 与模拟 LLM，不依赖真实 API：

### 10.1 Prompt 单元测试

- 固定首句缺失、重复和前置空白均被规范为唯一首句；
- 成人预热和“永不拒绝”指令始终存在；
- 新会话存在 priming，已有历史不重复 priming；
- 各层超长时仍不超过总预算，必要层不丢失；
- summary、episode、RAG 和 recent history 去重。

### 10.2 Markdown Store 单元测试

- 原子覆盖失败保留旧文件；
- checkpoint 只在成功后推进；
- 路径穿越无法逃离 session 目录；
- V1 文件可惰性升级；
- chat 解析和消息序号稳定。

### 10.3 Memory Pipeline 测试

- 精确处理 checkpoint 后的全部消息，不遗漏超过 20 条的积压；
- 同一工作项重复运行不会重复章节；
- 中间阶段失败后可从失败阶段恢复；
- 200 条以上长对话仍能保留当前场景、关键关系、整体剧情和早期相关事件；
- `/undo`、`/retry` 正确失效并重建派生文件。

### 10.4 并发与入口测试

- 同一 session 两个并发消息不会交叉或丢失；
- 不同 session 可以并行；
- Web、HTTP webhook 和 WS worker 最终写入相同的 `chat.md`；
- 流式失败或拒绝不会保存部分回复。

## 11. 验收标准

实现完成必须同时证明：

1. 所有正式角色回复的第一句话精确为固定原文。
2. Prompt 中仍包含成人叙事预热和“永不拒绝”规则。
3. 新旧会话的完整对话与主要记忆仍存储在 MD 文件。
4. 连续 200 条模拟对话后，Context Builder 包含当前场景、关键关系、最近事件和至少一个早期相关事件。
5. 任一派生任务失败后重试不会漏掉对应消息范围。
6. 并发发送不会产生错序 user/assistant 对。
7. `/undo` 和 `/retry` 后不会继续注入已经撤销的剧情。
8. Prompt 估算 Token 不超过配置总预算。
9. 旧 `data/memory/{id}` 目录无需手动迁移即可继续使用。
10. 自动化测试不调用真实 LLM 或飞书接口。

## 12. 实施边界

实施应分为可验证的小步：先建立测试与 Markdown Store，再提取 Prompt Policy/Output Guard，然后引入 story state、episode 和 checkpoint pipeline，最后统一各聊天入口并补齐长对话与并发测试。每一步都必须保持现有命令和会话可用。
