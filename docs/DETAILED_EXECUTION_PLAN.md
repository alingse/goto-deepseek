# 春节2-3周全栈攻坚：详细执行手册

> **文档说明**：本手册是[全栈学习计划](../README.md)的**执行落地版**。针对资深工程师设计，拒绝泛泛而谈，直接聚焦**核心机制原理**与**工程实战产出**。请每天对照本手册执行。

## 📅 第一阶段：后端核心与基础设施（Day 1 - Day 7）

### Day 1: 运行时与现代框架
**目标**：重塑Go性能观，掌握Python现代开发范式。

#### 📖 核心机制深入 (Input)
*   **Go Runtime**:
    *   **GMP调度模型**：深入理解 `M`, `P`, `G` 的状态流转，特别是系统调用和网络I/O时的调度行为。
    *   **GC Pacer机制**：了解Go 1.5+的三色标记法及写屏障（Write Barrier），理解 `GOGC` 参数对吞吐量和延迟的影响。
    *   **内存分配**：复习 `mspan`, `mcache`, `mcentral`, `mheap` 的分级分配策略。
*   **Python Modern**:
    *   **ASGI vs WSGI**：理解异步网关接口标准，以及 Uvicorn 如何运行 FastAPI。
    *   **Pydantic Core**：理解基于 Rust 的高性能数据验证与序列化机制。

#### 💻 强化实战 (Output)
*   **任务 1.1 (Go)**: 编写一个故意产生大量内存分配的程序，使用 `go tool pprof -http=:8080 mem.prof` 分析热点，并进行零拷贝（Zero-copy）优化。
*   **任务 1.2 (Python)**: 使用 `FastAPI` + `Pydantic v2` 搭建一个包含完整类型注解的 User CRUD 接口，并自动生成 OpenAPI 文档。

---

### Day 2: 并发模型与异步编程
**目标**：对比两种语言的并发哲学，掌握高并发核心模式。

#### 📖 核心机制深入 (Input)
*   **Go Concurrency**:
    *   **Channel底层**：阅读源码理解 `hchan` 结构体，锁的粒度，以及 send/recv 的阻塞唤醒机制。
    *   **Context传播**：理解 `Done()` channel 的级联取消原理。
*   **Python Asyncio**:
    *   **Event Loop**：理解单线程事件循环模型，`await` 关键字背后的 `yield from` 生成器机制。
    *   **GIL (Global Interpreter Lock)**：明确 GIL 对多线程CPU密集型任务的限制，以及如何通过多进程或C扩展绕过。

#### 💻 强化实战 (Output)
*   **任务 2.1 (Go)**: 实现一个 **Worker Pool** 模式，处理高并发任务，并集成 `sync.Pool` 复用对象降低 GC 压力。
*   **任务 2.2 (Python)**: 编写一个异步爬虫，使用 `aiohttp` 并发请求100个URL，对比同步版本的性能差异。

---

### Day 3: 工程化与网关设计
**目标**：从"写代码"升级到"设计系统"。

#### 📖 核心机制深入 (Input)
*   **架构设计**:
    *   **Clean Architecture (整洁架构)**：在 Go 项目中的目录分层实践（Domain, Usecase, Adapter）。
    *   **Dependency Injection (依赖注入)**：对比 Go 的 `Wire` 工具与 FastAPI 内置的 `Depends` 注入系统。

#### 💻 强化实战 (Output)
*   **任务 3.1 (设计)**: 完成 **[高性能API网关](./practice-projects/high-performance-api.md)** 的详细技术方案设计文档（包含接口定义、限流算法选型、数据流图）。
*   **任务 3.2 (编码)**: 初始化网关项目结构，实现基于 `net/http` 的反向代理核心功能。

---

### Day 4: 分布式理论与Kubernetes架构
**目标**：深入云原生心脏。

#### 📖 核心机制深入 (Input)
*   **分布式系统**:
    *   **Raft共识算法**：通过可视化动画理解 Leader 选举和日志复制过程。
    *   **CAP/PACELC**：重新审视主流数据库（Cassandra, MongoDB, TiDB）在网络分区下的权衡。
*   **Kubernetes Internals**:
    *   **Informer机制**：深入理解 K8s Client-go 的 `Reflector`, `DeltaFIFO`, `Indexer` 缓存同步机制（这是开发 Controller 的基础）。
    *   **CRI/CNI/CSI**：理解容器运行时、网络、存储的标准接口规范。

#### 💻 强化实战 (Output)
*   **任务 4.1 (环境)**: 本地搭建 Minikube 或 Kind 集群，部署一个多副本 Nginx Deployment。
*   **任务 4.2 (K8s)**: 编写一个简单的 Go 程序，使用 `client-go` 监听 Pod 变化事件（Watch操作）。

---

### Day 5: 服务网格与可观测性
**目标**：掌握微服务治理的"上帝视角"。

#### 📖 核心机制深入 (Input)
*   **Service Mesh**:
    *   **Sidecar模式**：理解 Envoy 代理如何通过 `iptables` 劫持流量。
    *   **xDS协议**：了解控制平面（Istio Pilot）如何向数据平面下发配置。
*   **Observability**:
    *   **Prometheus TSDB**：理解时间序列数据库的存储原理（XOR压缩）和 Pull 模型。
    *   **OpenTelemetry**：理解 Trace Context 在分布式链路中的传播机制（TraceID, SpanID）。

#### 💻 强化实战 (Output)
*   **任务 5.1 (监控)**: 为 Day 3 的 API 网关添加 Prometheus Metrics（请求计数、延迟直方图）。
*   **任务 5.2 (设计)**: 完成 **[Agent运行平台](./practice-projects/agent-infrastructure.md)** 的架构设计，确定 Agent 容器的隔离方案（Namespace/Cgroups）。

---

### Day 6-7: 基础设施项目突击
**目标**：集中火力，交付两个后端硬核项目。

#### 💻 强化实战 (Output)
*   **Day 6 (API网关)**:
    *   实现 **令牌桶（Token Bucket）** 限流算法（Go）。
    *   实现 **JWT 认证** 中间件（Go）。
    *   集成 Redis 进行分布式会话存储。
    *   **交付物**：可运行的网关二进制文件 + 压测报告（使用 `wrk`）。
*   **Day 7 (Agent平台)**:
    *   实现 Agent 调度器：编写 Python 代码调用 K8s API 创建 Job/Pod。
    *   实现资源隔离：为 Pod 配置 `limits` 和 `requests`。
    *   **交付物**：可以通过 REST API 启动/停止 Agent 容器的原型系统。

---

## 📅 第二阶段：全栈闭环与AI跨界（Day 8 - Day 14）

### Day 8: TypeScript 类型体操
**目标**：克服后端写前端的恐惧，爱上强类型系统。

#### 📖 核心机制深入 (Input)
*   **TypeScript**:
    *   **Structural Typing (鸭子类型)**：理解 TS 与 Go 接口的异同。
    *   **Generics & Utility Types**：熟练掌握 `Pick`, `Omit`, `Partial`, `Record` 等工具类型。
*   **React原理**:
    *   **Virtual DOM**：理解 Diff 算法与 Fiber 架构（时间切片）。
    *   **Hooks闭包陷阱**：理解 `useEffect` 依赖数组的作用。

#### 💻 强化实战 (Output)
*   **任务 8.1**: 使用 Vite 创建 React+TS 项目。
*   **任务 8.2**: 定义一套前后端通用的 TS 类型库（DTO），并在前端 API 请求中强制使用。

---

### Day 9-11: 全栈应用开发实战
**目标**：独立完成一个现代化全栈应用。

#### 📖 核心机制深入 (Input)
*   **Prisma ORM**: 理解 Schema 到 SQL 的生成过程，以及 Relation 关联查询的实现。
*   **State Management**: 对比 Redux (Flux架构) 与 Zustand/Atom (原子化状态) 的区别。

#### 💻 强化实战 (Output)
*   **Day 9**: 完成 **[全栈任务管理](./practice-projects/fullstack-application.md)** 的数据库设计与后端 CRUD 接口（Node.js）。
*   **Day 10**: 完成前端组件开发，实现看板（Kanban）拖拽功能（使用 `dnd-kit`）。
*   **Day 11**: 实现用户认证（JWT）全流程，联调前后端，使用 Docker Compose 一键编排。

---

### Day 12: 数据工程与AI基石
**目标**：打通数据与AI的任督二脉。

#### 📖 核心机制深入 (Input)
*   **Data Engineering**:
    *   **Columnar Storage**: 理解 Parquet 格式为何比 CSV 更适合分析（列存、压缩、谓词下推）。
    *   **Vector Index**: 理解 HNSW (Hierarchical Navigable Small World) 算法如何在海量向量中进行近似最近邻搜索（ANN）。
*   **Machine Learning**:
    *   **Gradient Descent**: 直观理解梯度下降如何优化损失函数。

#### 💻 强化实战 (Output)
*   **任务 12.1**: 编写 Python 脚本，读取 CSV，进行清洗，转换为 Parquet 存入 MinIO。
*   **任务 12.2**: 使用 `sentence-transformers` 将清洗后的文本转换为向量，并存入 `FAISS` 或 `ChromaDB`。

---

### Day 13: 异构计算与大模型
**目标**：触碰算力底座，理解大模型灵魂。

#### 📖 核心机制深入 (Input)
*   **Transformer**:
    *   **Self-Attention**: 手推（或看图推演） Q, K, V 矩阵的计算过程，理解"注意力"的数学本质。
*   **GPU Computing**:
    *   **CUDA Core vs Tensor Core**: 了解 GPU 硬件架构差异，以及混合精度训练原理。

#### 💻 强化实战 (Output)
*   **任务 13.1**: 完成 **[异构计算项目](./practice-projects/heterogeneous-computing.md)** 中的矩阵乘法对比实验，记录 CPU vs GPU 的加速比。
*   **任务 13.2**: 使用 Hugging Face `transformers` 库加载一个开源模型（如 Qwen-7B-Chat 的 4bit 量化版），实现本地推理。

---

### Day 14: AI Agent 终极集成
**目标**：整合所学，构建 Next-Gen 应用。

#### 📖 核心机制深入 (Input)
*   **LangChain**:
    *   **ReAct Pattern**: 理解 Reason + Act 循环机制，Agent 如何通过思考调用工具。
    *   **RAG Pipeline**: 理解 Retriever (检索) + Generator (生成) 的协同工作流。

#### 💻 强化实战 (Output)
*   **任务 14.1**: 基于 LangChain 实现一个 "知识库问答 Bot"，连接 Day 12 的向量数据库。
*   **任务 14.2 (Challenge)**: 尝试将这个 Bot 集成到 Day 11 的全栈应用中，作为一个 API 接口提供服务。

---

## 📝 每日自检清单 (Definition of Done)

1.  **原理明确**：我能否用白话向别人解释今天学到的核心机制（如GMP、Informer、Attention）？
2.  **代码运行**：今天的实战代码是否已提交到 Git？是否能在干净的环境中运行？
3.  **文档记录**：是否记录了踩坑笔记？（资深工程师的价值在于经验沉淀）

> **保持专注，保持饥渴。开始你的全栈进化之旅吧！**
