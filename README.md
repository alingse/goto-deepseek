# 全栈工程师学习计划（春节2-3周集中突破）

## 1. 概述

- **目标**：基于[fullstack-jd.md](./fullstack-jd.md)职位要求，系统提升全栈技能
- **时间窗口**：春节假期2-3周集中学习（每日4-6小时）
- **核心理念**：实践驱动、目标导向、时间盒约束
- **适用对象**：具有10年Go后端开发经验的资深工程师

## 2. 技能现状评估（简洁版）

| 技能领域 | 现状水平 | 与JD要求差距 | 结论与优先级 |
|----------|----------|--------------|--------------|
| **Go语言** | 精通（10年经验） | 需要复习运行时优化、最新特性 | ⭐⭐⭐⭐ 重点复习 |
| **Rust** | 未学过 | JD明确要求精通Rust/C++/TS/Python至少一门 | ⭐⭐⭐⭐ 建议学习 |
| **Python** | 曾经精通，需要更新 | 学习FastAPI、异步编程、现代生态 | ⭐⭐⭐ 系统学习 |
| **TypeScript** | 基础了解 | 需要系统学习前端开发、Node.js | ⭐⭐⭐⭐ 重点突破 |
| **分布式系统** | 精通（有经验） | 更新知识：服务网格、云原生架构 | ⭐⭐⭐ 巩固提升 |
| **数据库** | 精通（有经验） | 复习MySQL 8.0+、分布式事务 | ⭐⭐⭐ 巩固提升 |
| **数据工程** | 基础了解 | 学习ETL、数据清洗、数据湖 | ⭐⭐⭐ 专项补充 |
| **AI/ML** | 入门水平 | 系统性学习Transformer、大模型基础 | ⭐⭐⭐⭐⭐ 核心突破 |
| **云原生** | 熟悉基础 | 深入Kubernetes、可观测性、GitOps | ⭐⭐ 了解使用 |
| **系统原理** | 深刻理解 | 强化Profiling、性能分析能力 | ⭐⭐⭐ 实践提升 |

**核心结论**：
- **优势领域**：Go、分布式系统、数据库（需要复习更新）
- **重点突破**：AI/ML、TypeScript（从基础系统学习）
- **巩固提升**：Python现代化开发、系统原理实践
- **了解使用**：云原生、异构计算（结合工作实际）

## 3. 学习目标定制（基于JD全面覆盖）

### 3.1 高并发服务端与API系统（优先领域）
**JD要求**：数千万日活架构、性能优化、AI Chat Bot开发
**学习目标**：
1. 掌握FastAPI高性能API开发（Python现代化）
2. 深入Go语言运行时优化（Go复习）
3. 设计AI服务API架构（AI+Python实践）

**实践项目**：基于FastAPI和Go的混合架构AI服务网关

### 3.2 Agent基础设施与运行时平台（优先领域）
**JD要求**：容器调度隔离、资源管理、Agent运行时
**学习目标**：
1. 深入Kubernetes调度与网络策略（云原生进阶）
2. 理解容器安全隔离机制（系统原理）
3. 设计Agent沙箱环境（架构设计）

**实践项目**：简易Agent运行平台原型

### 3.3 异构超算基础设施（优先领域）
**JD要求**：GPU/NPU管理、集群调度、高性能计算
**学习目标**：
1. 了解GPU编程基础（CUDA/PyTorch）
2. 学习集群资源调度原理
3. 理解高性能计算通信模式

**实践项目**：学习笔记+概念验证代码

### 3.4 工程与架构能力突破
**JD要求**：语言能力、分布式系统、数据库
**学习目标**：
1. TypeScript全栈开发能力（重点突破）
2. 分布式系统设计模式复习
3. 数据库性能优化技巧更新

**实践项目**：全栈TypeScript应用 + 分布式事务Demo

### 3.5 系统与运维功底强化
**JD要求**：计算机原理、Profiling、Kubernetes
**学习目标**：
1. 深入Linux性能分析工具
2. Kubernetes运维最佳实践
3. 系统可观测性体系建设

**实践项目**：应用性能监控Dashboard

### 3.6 视野与思辨培养
**JD要求**：复杂系统探索、AGI思考、跨领域创新
**学习目标**：
1. 系统性学习AI/ML基础（Transformer为核心）
2. 跟踪AGI技术发展动态
3. 技术方案设计思维训练

**实践项目**：技术博客输出 + 架构设计文档

## 4. 详细学习路径（文档连接目录）

### 4.1 6个核心学习路径（独立文档）
1. **[Go语言深度复习](./docs/learning-paths/01-go-review.md)**
   - 重点：运行时优化、并发模式、性能调优
   - 时间：3天（春节第1周）

2. **[Python现代化开发](./docs/learning-paths/02-python-modern.md)**
   - 重点：FastAPI、异步编程、数据科学栈
   - 时间：3天（春节第1周）

3. **[TypeScript全栈学习](./docs/learning-paths/03-typescript-learning.md)**
   - 重点：类型系统、React前端、Node.js后端
   - 时间：4天（春节第2周）

4. **[AI/ML系统性学习](./docs/learning-paths/04-ai-ml-systematic.md)**
   - 重点：Transformer架构、大模型基础、LangChain
   - 时间：4天（春节第2周）

5. **[分布式系统复习](./docs/learning-paths/05-distributed-systems-review.md)**
   - 重点：微服务、消息队列、容错设计
   - 时间：2天（春节第1周）

6. **[云原生进阶](./docs/learning-paths/06-cloud-native-advanced.md)**
   - 重点：Kubernetes调度、可观测性、GitOps
   - 时间：2天（春节第1周）

7. **[大规模数据处理补充](./docs/learning-paths/07-data-engineering.md)** ⭐ 新增
   - 重点：ETL流程、数据质量、数据湖基础
   - 时间：穿插学习（第2周）

8. **[视野与思辨培养](./docs/learning-paths/08-vision-thinking.md)** ⭐ 新增
   - 重点：系统思维、AGI思考、跨领域创新
   - 时间：贯穿全程（每天晚上1-2小时）

9. **[Rust语言核心能力](./docs/learning-paths/09-rust-essentials.md)** ⭐ 新增
   - 重点：所有权系统、并发安全、零成本抽象
   - 时间：2-3天（可选/穿插进行）

### 4.2 4个实践项目（独立文档）
1. **[高性能API网关](./docs/practice-projects/high-performance-api.md)**
   - 技术栈：Go + FastAPI混合架构
   - 目标：实现限流、认证、监控功能

2. **[Agent运行平台原型](./docs/practice-projects/agent-infrastructure.md)**
   - 技术栈：Python + Kubernetes
   - 目标：简易Agent调度和隔离环境

3. **[异构计算学习项目](./docs/practice-projects/heterogeneous-computing.md)**
   - 技术栈：Python + PyTorch
   - 目标：GPU基础编程和性能分析

4. **[全栈TypeScript应用](./docs/practice-projects/fullstack-application.md)**
   - 技术栈：React + TypeScript + Node.js
   - 目标：完整的CRUD应用+用户认证

## 5. 时间安排（春节2-3周集中学习）

> **🚀 执行手册**：请务必查看 **[详细执行手册](./docs/DETAILED_EXECUTION_PLAN.md)**，这是你每天的行动指南，包含了核心机制解析和具体的编码任务。

> **日程表**：查看 **[每日详细日程表](./docs/schedule/daily-plan.md)** 获取时间分配建议。

### 第1周：基础巩固与重点突破（7天）
- **前3天**：Go复习 + Python现代化（并行）
- **中间2天**：分布式系统 + 云原生（并行）
- **后2天**：完成实践项目1+2

### 第2周：核心突破与实践（7天）
- **前4天**：TypeScript全栈学习（全天）
- **后3天**：AI/ML系统性学习（全天）
- **周末**：完成实践项目3+4

### 第3周：整合与总结（可选）
- 项目整合与优化
- 学习总结与文档整理
- 下一步学习计划制定

### 每日节奏（建议）
- **上午（3小时）**：理论学习 + 文档阅读
- **下午（3小时）**：编码实践 + 项目开发
- **晚上（1-2小时）**：总结反思 + 知识沉淀

## 6. 学习资源推荐（按优先级）

### 6.1 AI/ML资源（最高优先级）
- **入门必读**：《动手学深度学习》（李沐）
- **核心课程**：[吴恩达深度学习专项课程](https://www.coursera.org/specializations/deep-learning)
- **实践框架**：[LangChain官方文档](https://python.langchain.com/)
- **模型基础**：[Hugging Face Transformers教程](https://huggingface.co/learn)

### 6.2 TypeScript资源（高优先级）
- **官方文档**：[TypeScript Handbook](https://www.typescriptlang.org/docs/)
- **前端框架**：[React TypeScript指南](https://react-typescript-cheatsheet.netlify.app/)
- **全栈学习**：[Fullstack Open](https://fullstackopen.com/)
- **项目模板**：[Next.js + TypeScript](https://nextjs.org/docs)

### 6.3 Go语言资源（巩固复习）
- **最新特性**：[Go官方博客](https://go.dev/blog/)
- **性能优化**：[Dave Cheney博客](https://dave.cheney.net/)
- **实践模式**：[Go设计模式](https://github.com/tmrts/go-patterns)
- **并发编程**：《Go并发编程实战》

### 6.4 Python现代化资源
- **FastAPI**：[官方文档](https://fastapi.tiangolo.com/)
- **异步编程**：[Python asyncio指南](https://docs.python.org/3/library/asyncio.html)
- **数据科学**：[Pandas官方教程](https://pandas.pydata.org/docs/)
- **最佳实践**：[Real Python](https://realpython.com/)

### 6.5 分布式与云原生
- **分布式系统**：[MIT 6.824课程](https://pdos.csail.mit.edu/6.824/)
- **Kubernetes**：[K8s官方文档](https://kubernetes.io/docs/)
- **可观测性**：[OpenTelemetry](https://opentelemetry.io/)
- **架构设计**：《数据密集型应用系统设计》

## 7. 评估与调整机制

### 每周评估（周六进行）
1. **代码审查**：检查实践项目完成质量
2. **知识测试**：核心概念理解程度
3. **进度评估**：学习目标达成情况
4. **计划调整**：根据进度调整下周计划

### 关键成功指标
- ✅ 完成4个实践项目（可运行、有文档）
- ✅ 掌握9个学习路径核心概念
- ✅ 产出学习笔记和技术博客
- ✅ 建立个人知识库体系
- ✅ 形成对AGI的独立思考
- ✅ 掌握Rust基础编程能力

### 风险与应对
- **时间不足**：优先完成核心项目（AI服务+全栈应用）
- **概念困难**：调整学习顺序，先实践后理论
- **动力下降**：设定每日小目标，保持成就感

## 8. 重要提醒

### 春节学习窗口（2-3周）
- **机会难得**：集中时间突破技术瓶颈
- **务实可行**：目标明确，计划具体
- **实践导向**：编码时间占比60%以上
- **文档沉淀**：学习过程即知识积累

### 学习心态
- **接受不完美**：2-3周无法掌握所有，重点突破核心
- **享受过程**：技术探索本身有乐趣
- **保持灵活**：根据进度动态调整计划
- **成果导向**：以可展示的项目为目标

## 9. 开始学习

> **🚀 快速开始**：首次学习请先查看 **[快速开始指南](./QUICKSTART.md)**，5分钟快速启动学习环境。

### 第一步：环境准备（第1天）
1. 配置开发环境（Go、Python、Node.js）
2. 创建项目目录结构
3. 设置代码仓库和文档系统

### 第二步：按计划执行
1. 每天开始前阅读当日学习目标
2. 按时间块分配学习任务
3. 及时记录学习笔记和问题

### 第三步：持续改进
1. 每日总结学习收获
2. 每周评估调整计划
3. 完成项目后分享成果

---

**记住**：这个计划不是束缚，而是路线图。最重要的不是完美执行计划，而是在2-3周内实现实质性的技能提升。开始行动吧！

*计划版本：2026-02-08 v3.0*
*基于JD全面分析 + 技能现状评估 + 2-3周时间约束 + 新增Rust/视野思辨路径*