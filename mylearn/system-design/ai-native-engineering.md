# AI 原生工程化与自动化运维指南

本文档探讨史斌 CV 中的“AI Agent 个人探索”与“Vibe Coding”范式。这部分展示的是你作为资深工程师，如何利用 AI 彻底重塑传统的开发、调试与运维（DevOps）链路。

## 1. 核心理念深度拆解

### 1.1 文档即源码 (Documentation-as-Source)
*   **挑战**：代码生成虽然快，但难以维护。AI 经常理解错复杂的业务逻辑。
*   **细节**：
    *   **Spec-Driven**：不直接写代码，而是维护一份高精度的技术方案文档（Markdown）。
    *   **Context Control**：通过文档明确定义边界、接口和 SOP，让 AI 在受限的上下文中生成代码，确保生成的代码“符合架构预期”而不是天马行空。

### 1.2 MCP (Model Context Protocol) 基础设施
*   **细节**：如何让 AI 具备“手和脚”？
    *   **MCP Server**：将内部系统（Jenkins, GitLab, TiDB, Sentry）封装为标准 MCP 接口。
    *   **自动化诊断**：当线上 Sentry 报警时，On-call Agent 通过 MCP 自动抓取错误日志、最近的 Git Commit 和接口 Spec，生成修复建议并尝试跑通单元测试。

---

## 2. 专家级面试对垒

### 场景一：如何解决 AI 交付物的“黑盒”信任问题？
**面试官 (Expert)**：
> “你提倡用 AI 辅助甚至主导开发，但 AI 生成的代码可能存在逻辑漏洞或后门。作为资深架构师，你如何建立一套‘可验证性工程’来确保 AI 产出的质量？”

**候选人 (Senior Engineer)**：
> “我的核心原则是**‘Human-in-the-loop’与‘测试驱动的闭环验证’**：
> 1. **双向验证（Dual-Verification）**：AI 生成业务代码的同时，必须由另一个独立的 Agent（或人工）同步生成测试代码。代码合入的前提是 100% 覆盖率通过。
> 2. **静态扫描沙箱**：AI 产出的代码会自动送入 Lint、Security Scanner（如 Gosec）和内存泄露检测工具。
> 3. **影子运行（Shadow Mode）**：我之前在知乎做的‘策略预上线平台’。新代码合入后，先在生产环境‘只记录不动作’，将输出结果与旧版本对比。只有在数万次请求中表现一致或符合预期，才允许全量上线。”

### 场景二：基于 MCP 的自动化运维与风险控制
**面试官 (Expert)**：
> “你让 Agent 具备了操作 Jenkins 和 GitLab 的能力（On-call Agent）。这极具风险——如果 Agent 误删了生产数据库或触发了错误的发布流程怎么办？”

**候选人 (Senior Engineer)**：
> “这是**‘权限边界隔离’与‘原子化工具设计’**的问题：
> 1. **最小权限原则**：MCP 服务器本身不具备写权限，或者写权限被限制在极其狭窄的 API 内。例如，Agent 可以 `/check-diff-risk`，但不能直接 `/merge-and-deploy`。
> 2. **预定义动作集（Primitive Actions）**：我不给 Agent 提供 `bash` 权限，而是提供一组原子化的工具函数（如 `RestartService`, `Rollback`）。每个动作都内置了严格的校验逻辑。
> 3. **审批流集成**：在关键路径上引入‘人控开关’。Agent 生成诊断报告和修复建议，推送到 Slack/飞书，只有工程师点击 `Approve` 按钮，Agent 才会执行后续的物理发布。这种模式极大地降低了 On-call 的心理压力，同时也守住了底线。”

---

## 3. 针对 JD 的追问方向（自测）
1. **AI 辅助 Profiling**：如果让你设计一个 Agent，自动分析 `pprof` 图表并定位内存泄露，你会给它提供哪些 MCP 接口？
2. **知识库演进**：如何将线上的修复经验（Badcase 修复过程）自动沉淀为知识库，让下一任 Agent 变得更聪明？
3. **Prompt 的版本管理**：如何像管理代码一样管理复杂系统的 System Prompt？（引入 Prompt 单元测试与版本回滚机制）。
