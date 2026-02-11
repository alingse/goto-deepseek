# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 仓库概述

这是一个**全栈转型综合学习计划仓库**，为一位拥有10年Go后端经验的工程师在春节期间（2-3周集中学习）转型全栈而设计。所有内容完全基于具体的职位描述（参见 `fullstack-jd.md`），涵盖17个学习路径和4个实战项目。

**重要提示**：目前这是一个纯文档仓库，没有可执行代码。学习材料以Markdown参考文档的形式组织。

## 仓库结构

```
.
├── README.md                           # 学习计划总览
├── fullstack-jd.md                     # 目标职位描述（JD）
├── docs/
│   ├── DETAILED_EXECUTION_PLAN.md     # 每日执行详细指南
│   ├── learning-paths/                 # 17个专项学习路径
│   │   ├── LEARNING_PATHS_INDEX.md    # 导航与概览
│   │   ├── 01-go-review.md            # Go语言深度复习
│   │   ├── 02-python-modern.md        # Python现代化开发（FastAPI、异步）
│   │   ├── 03-typescript-learning.md  # TypeScript全栈学习
│   │   ├── 04-ai-ml-systematic.md     # AI/ML系统性学习
│   │   ├── 05-distributed-systems-review.md       # 分布式系统复习
│   │   ├── 06-cloud-native-advanced.md           # 云原生进阶
│   │   ├── 07-data-engineering.md                # 数据工程学习
│   │   ├── 08-vision-thinking.md                 # 视野与思辨培养
│   │   ├── 09-rust-essentials.md                 # Rust语言核心能力
│   │   ├── 10-distributed-storage-messaging.md   # 分布式存储与消息系统
│   │   ├── 11-network-programming.md             # 网络编程与高性能IO
│   │   ├── 12-algorithms-datastructures.md       # 算法与数据结构刷题
│   │   ├── 13-operating-systems.md               # 操作系统内核与原理
│   │   ├── 14-performance-profiling.md           # 系统性能优化与Profiling
│   │   ├── 15-computer-architecture.md           # 计算机组成与体系结构
│   │   ├── 16-search-vector-db.md                # 搜索引擎与向量数据库
│   │   └── 17-system-security.md                 # 系统安全基础
│   └── practice-projects/              # 4个实战项目
│       ├── high-performance-api.md     # 高性能API网关（Go + FastAPI）
│       ├── agent-infrastructure.md     # Agent运行平台（K8s）
│       ├── heterogeneous-computing.md  # 异构计算学习（GPU/PyTorch）
│       └── fullstack-application.md    # 全栈应用（React + Node.js）
└── mylearn/                            # 用户的个人学习工作区
```

## 核心学习理念

本仓库遵循**机制驱动、实践导向**的方法：

1. **JD映射**：每个学习主题都直接对应 `fullstack-jd.md` 中的要求
2. **核心机制聚焦**：强调深入理解底层原理（GMP调度器、Transformer注意力机制、Raft共识等）
3. **重实践目标**：60%以上的编码时间，有具体的交付物
4. **完成标准**：每天要求 - 能用通俗语言解释机制 + 代码可运行 + 有文档记录

## 核心JD要求（来自 `fullstack-jd.md`）

目标职位有4个职责领域：

1. **高并发服务端与API系统**：数千万日活、AI Chat Bot开发
2. **大规模数据处理Pipeline**：数据采集、清洗、数据湖、索引
3. **Agent基础设施与运行时平台**：容器调度、隔离、安全
4. **异构超算基础设施**：GPU/NPU管理、集群调度

## 学习路径依赖关系

```
基础层（路径 11-15）
    ↓
核心层（路径 5-6, 10, 14, 16-17）
    ↓
应用层（路径 1-4, 7, 9）

路径 08（视野与思辨）贯穿所有阶段
```

## 实战项目架构

### 1. 高性能API网关
- **技术栈**：Go（核心引擎）+ FastAPI（管理界面）+ Redis + Prometheus
- **核心功能**：限流（令牌桶）、JWT认证、路由、监控指标
- **学习目标**：混合架构，结合Go的高性能与Python的高生产力

### 2. Agent运行平台原型
- **技术栈**：Python + FastAPI + Kubernetes + Docker
- **核心功能**：容器调度、资源隔离（CPU/内存/GPU）、监控
- **学习目标**：理解K8s调度和容器安全机制

### 3. 异构计算项目
- **技术栈**：Python + PyTorch + CUDA
- **核心功能**：GPU编程基础、性能对比（CPU vs GPU）
- **学习目标**：为AI基础设施工作打基础

### 4. 全栈TypeScript应用
- **技术栈**：React + TypeScript + Node.js + Prisma
- **核心功能**：完整CRUD应用 + 认证 + 看板拖拽
- **学习目标**：完整的TypeScript前端 + 后端开发

## 在此仓库中工作

### 当用户要求开始学习时
1. 询问用户想从哪个学习路径或实战项目开始
2. 参考 `docs/DETAILED_EXECUTION_PLAN.md` 获取推荐的每日日程
3. 代码实现在 `mylearn/` 目录中进行（用户的个人工作区）

### 当用户想要实现实战项目时
1. 每个实战项目在其对应的 `.md` 文件中都有详细的实现指导
2. 项目包含具体的代码结构、测试策略和部署配置
3. 建议在实现之前在 `mylearn/` 中创建项目骨架

### 当用户需要JD说明时
直接参考 `fullstack-jd.md` - 它包含完整的职位描述，包括：
- 岗位职责（4个主要领域）
- 核心要求（工程/架构、系统/运维、视野/思辨）

### 当用户询问学习细节并要求保存时
如果用户开始询问不同方向的细节/知识并要求保存，将学习记录保存到 `mylearn/` 目录：

1. **文件命名格式**：`mylearn/{路径编号}-{路径简称}-{YYYYMMDD}-{主题简述}.md`
   - 示例：`mylearn/01-go-review-20260208-version-changes.md`
   - 示例：`mylearn/04-ai-ml-20260208-transformer-attention.md`
   - 示例：`mylearn/10-distributed-storage-20260208-redis-cluster.md`

2. **文件内容结构**：
   ```markdown
   # {主题标题}

   **日期**: {YYYY-MM-DD}
   **学习路径**: {路径编号} - {路径名称}
   **对话主题**: {具体主题}

   ## 问题背景
   {用户提出的问题或学习目标}

   ## 核心知识点
   {对话中讨论的核心概念和机制}

   ## 代码示例
   {如果有代码，记录在这里}

   ## 学习笔记
   {重要的理解、踩坑记录、参考资料等}

   ## 后续行动计划
   {接下来需要做的事情}
   ```

3. **保存时机**：
   - 用户明确要求"保存"或"记录下来"
   - 对话涉及重要的核心机制理解
   - 用户在探索新领域并积累了有价值的知识点
   - 完成了一个学习模块或实战任务

### 常用命令
由于这是纯文档仓库，常规开发命令（构建/测试/检查）尚不适用。实现实战项目时，请参考具体项目文档中针对特定技术的命令。

## AI/ML最新内容覆盖（2024-2025）

学习路径包含前沿AI/ML内容：
- **最新模型**：GPT-4o、Claude 3.5、Gemini 1.5、Llama 3.1/3.2、Qwen2.5、DeepSeek-V3
- **RAG 2.0与Agentic RAG**：高级检索技术、多Agent协作
- **新工具**：vLLM、Ollama、LangGraph、LlamaIndex 0.10+、Qdrant、Weaviate
- **AI安全**：RLHF、Constitutional AI、RLAIF

## 给AI助手的提示

1. **不要修改学习路径文档**，除非明确要求 - 这些是参考资料
2. **专注于实现**，当被要求帮助实战项目时
3. **使用 `mylearn/` 目录**进行任何代码生成或项目设置
4. **参考具体学习路径文档**，当问题涉及专门主题时
5. **用户是资深Go工程师**（10年经验） - 在帮助新领域时尊重他们的专业知识

## 学习理念总结

来自 `docs/DETAILED_EXECUTION_PLAN.md`：

> **每日自检清单**：
> 1. 我能否用白话向别人解释今天学到的核心机制？
> 2. 今天的实战代码是否已提交到Git？是否能在干净环境中运行？
> 3. 是否记录了踩坑笔记？（资深工程师的价值在于经验沉淀）

本仓库强调**机制理解胜过表面使用**和**实践实现胜过理论学习**。
