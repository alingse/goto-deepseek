# Kafka 深度解析与后端面试指南

**日期**: 2026-02-19  
**学习路径**: 10 - 分布式存储与消息系统  
**文档定位**: 适配全栈JD水准的 Kafka 深度知识 + 后端面试实战  

---

## 目录

1. [JD 要求映射与 Kafka 定位](#一jd-要求映射与-kafka-定位)
2. [Kafka 核心架构深度解析](#二kafka-核心架构深度解析)
3. [消息存储机制：从磁盘到内存](#三消息存储机制从磁盘到内存)
4. [生产者深度机制](#四生产者深度机制)
5. [消费者组与 Rebalance 协议](#五消费者组与-rebalance-协议)
6. [Exactly Once 语义实现](#六exactly-once-语义实现)
7. [高可用与 ISR 机制](#七高可用与-isr-机制)
8. [深度代码实战（Go + Java）](#八深度代码实战go--java)
9. [后端面试常见问题与答案](#九后端面试常见问题与答案)
10. [性能调优与生产实践](#十性能调优与生产实践)

---

## 一、JD 要求映射与 Kafka 定位

### 1.1 JD 职责领域对应

| JD 职责 | Kafka 相关能力 | 面试侧重点 |
|---------|---------------|-----------|
| **高并发服务端与 API 系统** | 消息队列削峰填谷、异步解耦 | 如何设计千万级日活的消息架构 |
| **大规模数据处理 Pipeline** | 数据采集、清洗、实时流处理 | Kafka Connect + Streams 实战 |
| **Agent 基础设施** | 事件驱动架构、日志聚合 | Kafka 作为事件总线设计 |
| **异构超算基础设施** | 高性能数据传输、低延迟消息 | Kafka 零拷贝与性能优化 |

### 1.2 核心能力要求拆解

```
JD 要求:"对分布式系统有深刻理解与实践经验"
    ↓
Kafka 考察点:
    ├── 分布式日志存储（Segment + Index）
    ├── 副本同步机制（ISR + HW + LEO）
    ├── 一致性协议（ZAB/KRaft）
    └── 分区与并行度设计

JD 要求:"负责核心服务的性能优化"
    ↓
Kafka 考察点:
    ├── 零拷贝技术（Zero-Copy）
    ├── 批量压缩与顺序写磁盘
    ├── 页缓存（Page Cache）利用
    └── 消费者并行度优化

JD 要求:"构建高质量数据湖与索引系统"
    ↓
Kafka 考察点:
    ├── CDC（变更数据捕获）设计
    ├── Kafka Connect 数据集成
    ├── 分层存储（Tiered Storage）
    └── 消息保留策略与 Compaction
```

---

## 二、Kafka 核心架构深度解析

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Kafka Cluster                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │  Broker 1   │    │  Broker 2   │    │  Broker 3   │                       │
│  │  (Leader)   │◄──►│  (Follower) │◄──►│  (Follower) │                       │
│  │             │    │             │    │             │                       │
│  │ Partition 0 │    │ Partition 1 │    │ Partition 2 │                       │
│  │ Partition 1 │    │ Partition 2 │    │ Partition 0 │                       │
│  └─────────────┘    └─────────────┘    └─────────────┘                       │
│         ▲                  ▲                  ▲                              │
└─────────┼──────────────────┼──────────────────┼──────────────────────────────┘
          │                  │                  │
    ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
    │ Producer  │      │ Consumer  │      │ Consumer  │
    │ (订单服务) │      │  Group A  │      │  Group B  │
    └───────────┘      │ (3实例)   │      │ (5实例)   │
                       └───────────┘      └───────────┘
```

### 2.2 核心概念详解

| 概念 | 定义 | 关键机制 | 面试常见问题 |
|------|------|---------|-------------|
| **Topic** | 逻辑消息分类 | 分区（Partition）实现水平扩展 | 为什么分区数不能随意增加？ |
| **Partition** | 物理存储单元 | 顺序写、不可变、分段存储 | 如何保证分区内的消息顺序？ |
| **Broker** | Kafka 服务器节点 | 负责消息存储与转发 | Broker 挂了会怎样？ |
| **Replica** | 分区副本 | Leader + Follower 机制 | ISR 是什么？为什么重要？ |
| **Offset** | 消息在分区中的唯一标识 | 消费者位移管理 | 提交 Offset 的时机选择？ |
| **Consumer Group** | 消费者逻辑组 | 组内负载均衡、Rebalance | Rebalance 抖动如何优化？ |

### 2.3 元数据管理演进：ZooKeeper vs KRaft

```
【Kafka < 3.3】ZooKeeper 模式
    ┌──────────────────────────────────────┐
    │           ZooKeeper Ensemble         │
    │    (管理 Broker 注册、Topic 元数据)   │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────┼───────────────────┐
    ▼                  ▼                   ▼
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Broker  │◄────►│ Broker  │◄────►│ Broker  │
│  (Controller)   │         │      │         │
└─────────┘      └─────────┘      └─────────┘

问题：
1. 外部依赖，运维复杂
2. ZK 成为性能瓶颈（大规模集群）
3. 元数据不一致风险（脑裂）

【Kafka >= 3.3】KRaft 模式（Kafka Raft Metadata）
    ┌──────────────────────────────────────┐
    │        KRaft Quorum (3 Controllers)   │
    │    (内部管理元数据，基于 Raft 协议)     │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────┼───────────────────┐
    ▼                  ▼                   ▼
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Broker  │      │ Broker  │      │ Broker  │
│         │      │         │      │         │
└─────────┘      └─────────┘      └─────────┘

优势：
1. 去除 ZK 依赖，简化架构
2. 元数据变更效率提升 10x+
3. 单集群可支持百万分区
4. 避免 ZK 会话超时导致的抖动
```

---

## 三、消息存储机制：从磁盘到内存

### 3.1 存储层次结构

```
Topic: orders
└── Partition 0 (目录: orders-0/)
    ├── 00000000000000000000.log      # 消息数据文件
    ├── 00000000000000000000.index    # 稀疏索引（Offset -> Position）
    ├── 00000000000000000000.timeindex # 时间索引（Timestamp -> Offset）
    ├── 00000000000356892104.log      # 下一个 Segment
    ├── 00000000000356892104.index
    └── 00000000000356892104.timeindex

Segment 文件命名规则：起始 Offset（20位，不足补零）
```

### 3.2 消息格式深度解析（V2 版本）

```
【Record Batch 结构】批量消息存储（提升吞吐）

┌─────────────────────────────────────────────────────────────┐
│                    Record Batch Header                       │
├─────────────────────────────────────────────────────────────┤
│ Base Offset          │ 8 bytes │ 批次第一条消息的 Offset     │
│ Batch Length         │ 4 bytes │ 批次总长度                  │
│ Partition Leader Epoch│ 4 bytes │ Leader 纪元号（防止数据回环）│
│ Magic                │ 1 byte  │ 版本号（V2 = 2）            │
│ CRC                  │ 4 bytes │ 校验和                      │
│ Attributes           │ 2 bytes │ 压缩类型、事务标志等          │
│ Last Offset Delta    │ 4 bytes │ 批次最后一条相对 Offset     │
│ First Timestamp      │ 8 bytes │ 批次第一条消息时间戳          │
│ Max Timestamp        │ 8 bytes │ 批次最大时间戳               │
│ Producer ID          │ 8 bytes │ 幂等生产者唯一 ID            │
│ Producer Epoch       │ 2 bytes │ 生产者纪元                   │
│ Base Sequence        │ 4 bytes │ 批次起始序列号               │
│ Records Count        │ 4 bytes │ 消息条数                     │
├─────────────────────────────────────────────────────────────┤
│                      Records[]                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Record 1: [Length][Attributes][Timestamp Delta][Offset  │ │
│  │ Delta][Key Len][Key][Value Len][Value][Headers[]]      │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Record 2: ...                                           │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

关键设计：
1. 批量存储减少磁盘 I/O
2. 相对 Offset 和 Timestamp 减少存储（Varint 编码）
3. 零拷贝传输时需要对齐页边界（4KB）
```

### 3.3 稀疏索引机制

```
【索引查找流程】查找 Offset = 3689214 的消息

1. 文件名二分查找定位 Segment
   - 00000000000000000000.log (Offset 0 ~ 3568921)
   - 00000000000356892104.log (Offset 356892104 ~ ...)
   → 目标在第二个 Segment

2. 加载 .index 文件到内存（稀疏索引）
   
   Index Entry 结构（每个 entry 8 bytes）：
   ┌───────────────┬───────────────┐
   │ Relative Offset│ Physical Pos  │
   │   (4 bytes)    │  (4 bytes)   │
   └───────────────┴───────────────┘
   
   稀疏索引：每写入一定字节（默认 4KB）才写入一条索引
   
   Offset: 356892104 → Position: 0
   Offset: 356892231 → Position: 10240    ← 每 4096 字节一条
   Offset: 356892358 → Position: 20480
   ...
   Offset: 3689210   → Position: 567890

3. 内存二分查找定位最近索引
   - 找到 Offset 3689210 在 Position 567890

4. 从 Position 567890 开始顺序扫描 .log 文件
   - 找到 Offset 3689214 的消息

时间复杂度：O(log N) 定位 Segment + O(log M) 索引 + O(K) 顺序扫描
```

### 3.4 零拷贝（Zero-Copy）技术深度解析

```
【传统文件读取 vs 零拷贝】

传统方式（4次拷贝，4次上下文切换）：
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  磁盘    │───►│ Page Cache│───►│ 应用缓冲区│───►│ Socket  │───►│ 网络 NIC │
│          │    │ (内核态)  │    │ (用户态)  │    │ 缓冲区   │    │         │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     DMA Copy      CPU Copy        CPU Copy        DMA Copy

零拷贝 - sendfile()（2次拷贝，2次上下文切换）：
┌──────────┐    ┌──────────┐    ┌──────────┐
│  磁盘    │───►│ Page Cache│───►│ 网络 NIC │
│          │    │ (内核态)  │    │         │
└──────────┘    └──────────┘    └──────────┘
     DMA Copy                     DMA Copy
     
零拷贝 - sendfile() + DMA Gather（1次拷贝，2次上下文切换）：
┌──────────┐    ┌────────────────────────────────────┐
│  磁盘    │───►│ Page Cache ───────────────────────►│───► 网络 NIC
│          │    │   ▲ 描述符（地址+长度）              │     DMA 直接从 Page Cache 读取
└──────────┘    └───┴────────────────────────────────┘

【Kafka 中的实现】

Kafka 使用 Java NIO 的 FileChannel.transferTo()：

// Java 伪代码
FileChannel fileChannel = new RandomAccessFile(logFile, "r").getChannel();
SocketChannel socketChannel = SocketChannel.open();

// 核心：零拷贝传输
fileChannel.transferTo(position, count, socketChannel);

// 底层调用 Linux sendfile() 系统调用
// ssize_t sendfile(int out_fd, int in_fd, off_t *offset, size_t count);

性能提升：
- 传统方式：约 65MB/s
- 零拷贝：约 700MB/s+（10x 提升）
```

### 3.5 页缓存（Page Cache）与刷盘策略

```
【Page Cache 利用策略】

生产者 ──► 消息 ──► 写入 Page Cache ──► 异步刷盘 ──► 磁盘
                         │
                         └──► 消费者读取（直接从 Page Cache，无需磁盘 I/O）

关键配置：
┌─────────────────────────────────────────────────────────────────────────┐
│ log.flush.interval.messages = 10000    # 累计 10000 条消息刷盘          │
│ log.flush.interval.ms = 1000           # 每秒刷盘一次                   │
│ log.segment.bytes = 1073741824         # Segment 大小 1GB               │
│ log.roll.hours = 168                   # 7 天滚动新 Segment             │
└─────────────────────────────────────────────────────────────────────────┘

刷盘策略对比：
┌─────────────┬─────────────┬─────────────┬──────────────────────────────┐
│   策略      │   可靠性    │   性能      │         适用场景              │
├─────────────┼─────────────┼─────────────┼──────────────────────────────┤
│ 异步刷盘    │ 较低        │ 最高        │ 日志、监控、允许少量丢失       │
│ 间隔刷盘    │ 中等        │ 高          │ 普通业务消息                  │
│ 同步刷盘    │ 最高        │ 低          │ 金融交易（不推荐用 Kafka）    │
└─────────────┴─────────────┴─────────────┴──────────────────────────────┘
```

---

## 四、生产者深度机制

### 4.1 生产者架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Kafka Producer                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐                                              │
│  │  Producer API │  send(ProducerRecord)                        │
│  │   (用户线程)   │                                              │
│  └───────┬───────┘                                              │
│          ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Record Accumulator                        ││
│  │  (内存缓冲区，默认 32MB)                                       ││
│  │  ┌───────────────────────────────────────────────────────┐  ││
│  │  │ Deque<RecordBatch> (Topic-Partition 队列)              │  ││
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │  ││
│  │  │  │ Batch 1  │  │ Batch 2  │  │ Batch 3  │             │  ││
│  │  │  │ (未填满) │  │ (未填满) │  │ (已填满) │             │  ││
│  │  │  └──────────┘  └──────────┘  └──────────┘             │  ││
│  │  └───────────────────────────────────────────────────────┘  ││
│  └───────────────┬──────────────────────────────────────────────┘│
│                  │ Sender 线程（后台）                           │
│                  ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      Network Client                          ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    ││
│  │  │ InFlightRequest│  │ InFlightRequest│  │ InFlightRequest│   ││
│  │  │ (Broker 1)     │  │ (Broker 2)     │  │ (Broker 3)     │   ││
│  │  │ max.in.flight  │  │               │  │               │    ││
│  │  │ = 5 (默认)     │  │               │  │               │    ││
│  │  └───────────────┘  └───────────────┘  └───────────────┘    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 生产者关键配置深度解析

```go
// Go 生产者配置示例（sarama 库）
config := sarama.NewConfig()

// ========== 批处理与压缩 ==========
// 批量发送大小（字节），默认 16384（16KB）
// 增大可提升吞吐，但会增加延迟
config.Producer.Flush.Bytes = 65536  // 64KB

// 批量等待时间，默认 0（立即发送）
// 与 Flush.Bytes 是"或"关系，满足任一条件即发送
config.Producer.Flush.Frequency = 10 * time.Millisecond

// 批量最大消息数，默认 0（不限制）
config.Producer.Flush.MaxMessages = 100

// 压缩类型：none, gzip, snappy, lz4, zstd
// 推荐 lz4（CPU/压缩比平衡）或 zstd（更高压缩比）
config.Producer.Compression = sarama.CompressionLZ4

// ========== 可靠性配置 ==========
// acks 配置：
// 0 - 不等待确认（最高吞吐，可能丢失）
// 1 - 等待 Leader 确认（平衡）
// all - 等待所有 ISR 确认（最高可靠）
config.Producer.RequiredAcks = sarama.WaitForAll

// 重试次数（默认 3）
config.Producer.Retry.Max = 5

// 重试间隔（默认 100ms）
config.Producer.Retry.Backoff = 200 * time.Millisecond

// 启用幂等性（自动设置 acks=all, retries=MAX_INT, max.in.flight=5）
config.Producer.Idempotent = true

// ========== 缓冲区与超时 ==========
// 发送超时
config.Producer.Timeout = 30 * time.Second

// 元数据刷新间隔
config.Metadata.RefreshFrequency = 5 * time.Minute

// 客户端 ID（用于监控识别）
config.ClientID = "order-service-producer"
```

### 4.3 消息分区策略

```go
// Go 分区策略实现

// 1. 轮询策略（RoundRobin）- 默认无 Key 时使用
func RoundRobinPartitioner(topic string, key []byte, value []byte, 
    partitionCount int32, partitions []int32) int32 {
    // 轮询选择分区，保证均匀分布
    // 缺点：无法保证消息顺序
    return partitions[atomic.AddInt32(&counter, 1)%int32(len(partitions))]
}

// 2. Key Hash 策略 - 有 Key 时默认使用
func HashPartitioner(topic string, key []byte, value []byte,
    partitionCount int32, partitions []int32) int32 {
    // murmur2 hash（Kafka 默认 hash 算法）
    hash := murmur2(key)
    // 取模确定分区
    return partitions[hash%uint32(len(partitions))]
}

// 3. 自定义分区策略 - 业务维度分区（保证业务内顺序）
func CustomPartitioner(topic string, key []byte, value []byte,
    partitionCount int32, partitions []int32) int32 {
    // 假设 key 是 orderID，提取用户 ID 作为分区依据
    // 保证同一用户的订单进入同一分区（用户内有序）
    userID := extractUserID(key)
    return partitions[userID%uint32(len(partitions))]
}

// 分区选择建议：
// ┌─────────────────────────────────────────────────────────────────┐
// │ 场景                    │ 推荐策略    │ 说明                    │
// ├─────────────────────────────────────────────────────────────────┤
// │ 无顺序要求，最大化吞吐   │ 轮询        │ 均匀分布，并行消费       │
// │ 全局顺序要求            │ 单分区      │ 牺牲吞吐，保证顺序       │
// │ 业务内顺序（如用户订单） │ Key Hash    │ 业务并行，业务内有序     │
// │ 地域/机房感知           │ 自定义      │ 就近生产消费             │
// └─────────────────────────────────────────────────────────────────┘
```

### 4.4 幂等性与事务消息

```
【幂等性 Producer 实现原理】

场景：网络超时导致 Producer 重试，可能产生重复消息

解决方案（PID + Sequence Number）：
┌─────────────────────────────────────────────────────────────────┐
│                     Broker 端幂等性检查                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  生产者 A (PID=1001, Epoch=0)                                    │
│     │                                                           │
│     ├──► Send Batch (Seq=0~9) ──► Broker                        │
│     │                              │                            │
│     │                              ▼                            │
│     │                         ┌─────────┐                       │
│     │                         │ 缓存 PID 1001 的最新 Seq         │
│     │                         │ lastSeq = 9                     │
│     │                         └─────────┘                       │
│     │                                                           │
│     ├──► 网络超时（实际 Broker 已写入）                          │
│     │                                                           │
│     └──► 重试 Send Batch (Seq=0~9) ──► Broker                   │
│                                        │                        │
│                                        ▼                        │
│                                   检查 Seq：                    │
│                                   - 若 Seq < lastSeq：重复消息，直接返回 OK │
│                                   - 若 Seq == lastSeq+1：新消息，正常写入   │
│                                   - 若 Seq > lastSeq+1：消息丢失，报错      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

限制：
1. 仅保证单分区单会话幂等（PID 重启会变化）
2. 需要配合事务才能实现跨分区幂等
```

```go
// 事务消息 Go 示例
func transactionalProducer() {
    config := sarama.NewConfig()
    config.Producer.Idempotent = true  // 启用幂等
    config.Producer.Transaction.ID = "order-tx-producer-1"  // 事务 ID
    
    producer, _ := sarama.NewSyncProducer(brokers, config)
    defer producer.Close()
    
    // 开启事务
    _ = producer.BeginTxn()
    
    // 发送多条消息（原子性）
    producer.SendMessage(&sarama.ProducerMessage{
        Topic: "orders",
        Key:   sarama.StringEncoder("order-001"),
        Value: sarama.StringEncoder(`{"amount": 100}`),
    })
    
    producer.SendMessage(&sarama.ProducerMessage{
        Topic: "inventory",
        Key:   sarama.StringEncoder("sku-123"),
        Value: sarama.StringEncoder(`{"delta": -1}`),
    })
    
    // 提交事务（或 AbortTxn() 回滚）
    _ = producer.CommitTxn()
}

// 事务保证：
// - 跨分区原子写入：orders 和 inventory 都成功或都失败
// - 消费-转换-生产（Consume-Transform-Produce）模式的原子性
// - 配合隔离级别 read_committed，消费者不读取未提交消息
```

---

## 五、消费者组与 Rebalance 协议

### 5.1 消费者组架构

```
Topic: orders (6 个分区)

Consumer Group: order-processors (3 个实例)
┌─────────────────────────────────────────────────────────────────┐
│                     Group Coordinator                           │
│                      (Broker 负责管理)                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Group State: Stable                                      │    │
│  │ Members:                                                 │    │
│  │   - consumer-1 (instance-1): partitions [0, 1]          │    │
│  │   - consumer-2 (instance-2): partitions [2, 3]          │    │
│  │   - consumer-3 (instance-3): partitions [4, 5]          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

分区分配策略：
1. Range（默认）：按分区范围分配
   - C1: [0, 1], C2: [2, 3], C3: [4, 5]
   - 问题：分区数不整除时分配不均

2. RoundRobin：轮询分配
   - C1: [0, 3], C2: [1, 4], C3: [2, 5]
   - 更均匀，但不保证顺序相邻

3. Sticky（推荐）：尽量保持分配稳定
   - 新增消费者时，仅移动必要分区
   - 减少 Rebalance 带来的位移丢失
   - Kafka 2.4+ 默认使用

4. Cooperative Sticky（Kafka 2.4+）：
   - 增量式 Rebalance，不停消费
   - 先撤销部分分区，再重新分配
   - 极大减少 Rebalance 停顿时间
```

### 5.2 Rebalance 触发与过程

```
【Rebalance 触发条件】

1. 新消费者加入组
2. 消费者退出组（正常关闭或超时）
3. 消费者心跳超时（session.timeout.ms）
4. 分区数增加（Topic 扩容）
5. 订阅的 Topic 元数据变化

【Rebalance 过程】（Eager 模式）

Consumer-1         Consumer-2         Coordinator
     │                  │                  │
     │  JoinGroup       │                  │
     ├─────────────────►│                  │
     │                  │  JoinGroup       │
     │                  ├─────────────────►│
     │                  │                  │
     │                  │  SyncGroup       │
     │◄─────────────────┼──────────────────┤
     │  分配方案 [0,1]   │  分配方案 [2,3]  │
     │                  │                  │
     │◄─────────────────┼──────────────────┤
     │  停止消费，提交 Offset                        │
     │                  │                  │
     │  开始消费 [0,1]   │  开始消费 [2,3]  │

问题：
- Rebalance 期间全组停止消费
- 分区越多、消费者越多，停顿越久（可达数秒）

【优化：Cooperative Rebalance】

Consumer-1         Consumer-2         Coordinator
     │                  │                  │
     │  JoinGroup       │                  │
     ├─────────────────►│                  │
     │                  │                  │
     │                  │  Revoke [3]      │  ← 只撤销需要迁移的分区
     │◄─────────────────┼──────────────────┤
     │                  │  继续消费 [2]     │  ← 其他分区不停！
     │                  │                  │
     │  增量分配 [3]     │                  │
     │◄─────────────────┼──────────────────┤
     │  开始消费 [0,1,3] │                  │
```

### 5.3 位移提交策略

```go
// Go 消费者位移提交策略

// 策略 1：自动提交（默认，可能导致重复消费或消息丢失）
config := sarama.NewConfig()
config.Consumer.Offsets.AutoCommit.Enable = true
config.Consumer.Offsets.AutoCommit.Interval = 1 * time.Second

// 策略 2：手动同步提交（推荐，精确控制）
func manualCommit(consumer sarama.ConsumerGroup) {
    for msg := range claim.Messages() {
        // 处理消息
        if err := process(msg); err != nil {
            // 处理失败，不提交，下次重试
            continue
        }
        // 处理成功，立即提交
        consumer.Commit()
    }
}

// 策略 3：手动异步提交（高性能场景）
func asyncCommit(consumer sarama.ConsumerGroup) {
    commitCh := make(chan *sarama.ConsumerMessage, 100)
    
    // 批量提交协程
    go func() {
        ticker := time.NewTicker(5 * time.Second)
        defer ticker.Stop()
        
        var pending []*sarama.ConsumerMessage
        for {
            select {
            case msg := <-commitCh:
                pending = append(pending, msg)
                if len(pending) >= 100 {
                    commitBatch(consumer, pending)
                    pending = pending[:0]
                }
            case <-ticker.C:
                if len(pending) > 0 {
                    commitBatch(consumer, pending)
                    pending = pending[:0]
                }
            }
        }
    }()
}

// 策略 4：精准一次（事务消费）
func exactlyOnce(consumer sarama.ConsumerGroup, producer sarama.SyncProducer) {
    for msg := range claim.Messages() {
        // 开启消费事务
        _ = producer.BeginTxn()
        
        // 发送下游消息
        producer.SendMessage(transform(msg))
        
        // 提交消费位移 + 发送消息原子性
        _ = producer.SendOffsetsToTxn(
            map[sarama.TopicPartition]int64{
                {Topic: msg.Topic, Partition: msg.Partition}: msg.Offset + 1,
            },
            consumerGroupID,
        )
        
        // 提交事务
        _ = producer.CommitTxn()
    }
}

// 位移提交位置选择：
// ┌───────────────────────────────────────────────────────────────────┐
// │ 位置                │ 优点                │ 缺点                  │
// ├───────────────────────────────────────────────────────────────────┤
// │ 处理前提交          │ 不重复消费          │ 可能丢失消息          │
// │ 处理后提交（推荐）   │ 不丢失消息          │ 可能重复消费          │
// │ 事务提交            │ 精确一次            │ 性能较低，复杂度高    │
// └───────────────────────────────────────────────────────────────────┘
```

---

## 六、Exactly Once 语义实现

### 6.1 三种语义对比

```
┌──────────────────────────────────────────────────────────────────────┐
│                    消息投递语义对比                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  At Most Once（最多一次）                                             │
│  ┌──────────┐                                                         │
│  │ Producer │────► 消息 ────► Consumer ──► 立即提交 Offset            │
│  └──────────┘                                 │                       │
│                                               ▼                       │
│                                          处理消息                      │
│  特点：可能丢失，不会重复                                             │
│  场景：日志、监控（可容忍丢失）                                       │
│                                                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  At Least Once（至少一次）                                            │
│  ┌──────────┐                                                         │
│  │ Producer │────► 消息 ────► Consumer ──► 处理消息                   │
│  └──────────┘                                 │                       │
│                                               ▼                       │
│                                          提交 Offset                   │
│  特点：不会丢失，可能重复                                             │
│  场景：大多数业务场景（需幂等消费）                                   │
│                                                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Exactly Once（精确一次）                                             │
│  ┌──────────┐                                                         │
│  │ Producer │────► 事务消息 ────► Consumer                            │
│  │ (幂等+事务)│               (隔离级别=read_committed)                │
│  └──────────┘         │                                               │
│                       ▼                                               │
│                  ┌─────────┐                                          │
│                  │ 事务协调器 │                                        │
│                  │ 原子性保证 │                                        │
│                  └─────────┘                                          │
│  特点：不丢失，不重复                                                 │
│  场景：金融交易、对账场景                                             │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 精确一次实现架构

```
【跨系统 Exactly Once 方案】

场景：MySQL → Kafka → Consumer → MySQL

方案 1：两阶段提交（2PC）- 不推荐
- 强依赖 XA 事务，性能差
- 单点协调器问题

方案 2：本地消息表（推荐）
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│  业务 MySQL   │        │    Kafka     │        │  下游 MySQL   │
├──────────────┤        ├──────────────┤        ├──────────────┤
│ 业务表       │        │              │        │              │
│ 本地消息表   │───────►│ Topic        │───────►│ 业务表       │
│ (待发送)     │        │              │        │              │
└──────────────┘        └──────────────┘        └──────────────┘
       │
       ▼
   定时扫描发送
   成功更新状态

方案 3：CDC + 幂等消费
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│  业务 MySQL   │──CDC──►│    Kafka     │───────►│  下游 MySQL   │
│ (Binlog)     │        │              │ 幂等   │              │
└──────────────┘        └──────────────┘        └──────────────┘
- 天然记录所有变更
- 消费者基于主键幂等写入
```

### 6.3 幂等消费实现代码

```go
// 幂等消费实现（基于 Redis 去重）

package consumer

import (
    "context"
    "fmt"
    "time"
    
    "github.com/IBM/sarama"
    "github.com/redis/go-redis/v9"
)

type IdempotentConsumer struct {
    redis      *redis.Client
    processor  MessageProcessor
    dedupTTL   time.Duration
}

// 幂等键格式: kafka:dedup:{topic}:{partition}:{offset}
func (c *IdempotentConsumer) ConsumeClaim(
    sess sarama.ConsumerGroupSession, 
    claim sarama.ConsumerGroupClaim,
) error {
    for msg := range claim.Messages() {
        dedupKey := fmt.Sprintf("kafka:dedup:%s:%d:%d", 
            msg.Topic, msg.Partition, msg.Offset)
        
        // 1. 检查是否已处理（Redis SETNX）
        set, err := c.redis.SetNX(context.Background(), dedupKey, "1", c.dedupTTL).Result()
        if err != nil {
            log.Error("dedup check failed", err)
            continue  // 不提交，下次重试
        }
        
        if !set {
            // 已处理过，直接提交 Offset
            log.Info("duplicate message skipped", 
                "topic", msg.Topic, 
                "partition", msg.Partition, 
                "offset", msg.Offset)
            sess.MarkMessage(msg, "")
            continue
        }
        
        // 2. 业务处理
        if err := c.processor.Process(msg); err != nil {
            // 处理失败，删除去重标记，允许重试
            c.redis.Del(context.Background(), dedupKey)
            log.Error("process failed", err)
            continue
        }
        
        // 3. 提交 Offset
        sess.MarkMessage(msg, "")
    }
    return nil
}

// 另一种幂等方式：业务唯一键去重
type OrderConsumer struct {
    db         *gorm.DB
    idempotencyKey string  // 消息中的唯一键，如 orderID
}

func (c *OrderConsumer) Process(msg *sarama.ConsumerMessage) error {
    var event OrderEvent
    if err := json.Unmarshal(msg.Value, &event); err != nil {
        return err
    }
    
    // 使用数据库唯一索引保证幂等
    return c.db.Transaction(func(tx *gorm.DB) error {
        // INSERT IGNORE / ON DUPLICATE KEY UPDATE
        result := tx.Exec(`
            INSERT INTO order_events (idempotency_key, order_id, status, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (idempotency_key) DO NOTHING
        `, event.OrderID+"_"+event.EventType, event.OrderID, event.Status, time.Now())
        
        if result.RowsAffected == 0 {
            // 重复消息，忽略
            return nil
        }
        
        // 执行业务逻辑
        return c.updateOrder(tx, event)
    })
}
```

---

## 七、高可用与 ISR 机制

### 7.1 副本同步机制

```
【ISR（In-Sync Replicas）机制】

Topic: orders, Partition 0, Replication Factor = 3

┌─────────────────────────────────────────────────────────────────┐
│                     Partition 0                                  │
│                                                                  │
│   ┌───────────┐        ┌───────────┐        ┌───────────┐       │
│   │  Replica 0 │        │  Replica 1 │        │  Replica 2 │       │
│   │  (Leader)  │◄──────►│ (Follower)│◄──────►│ (Follower)│       │
│   │            │ 同步   │            │ 同步   │            │       │
│   │   HW=100   │        │   HW=100   │        │   HW=80    │       │
│   │   LEO=120  │        │   LEO=120  │        │   LEO=100  │       │
│   └───────────┘        └───────────┘        └───────────┘       │
│        ▲                      │                      │          │
│        │                      └──────────────────────┘          │
│        │                         ISR = {0, 1}                    │
│        │                      (Replica 2 延迟太大，踢出 ISR)      │
│   Producer                                                      │
└─────────────────────────────────────────────────────────────────┘

关键概念：
- HW (High Watermark): 消费者可见的最大 Offset，ISR 中最小的 LEO
- LEO (Log End Offset): 副本的日志末尾 Offset
- ISR: 与 Leader 保持同步的副本集合

写入流程：
1. Producer 发送消息到 Leader
2. Leader 写入本地 Log，LEO = 121
3. Follower 拉取消息并写入
4. Leader 检查 Follower LEO，更新 HW
5. 当 acks=all，等待所有 ISR 确认后才响应 Producer
```

### 7.2 Leader 选举与数据一致性

```
【Leader 故障场景】

场景：Leader (Replica 0) 宕机，ISR = {0, 1, 2}

情况 1：ISR 中有存活副本
- 从 ISR 中选择新 Leader（通常是 LEO 最大的）
- 原 Leader 恢复后成为 Follower
- 数据一致性保证

情况 2：所有 ISR 都宕机（极端情况）
策略选择（unclean.leader.election.enable）：

┌─────────────────────────────────────────────────────────────────┐
│ unclean.leader.election.enable = false（默认，推荐）             │
├─────────────────────────────────────────────────────────────────┤
│ 行为：等待 ISR 中任意副本恢复，在此期间该分区不可写               │
│                                                                  │
│ 优点：数据不丢失                                                 │
│ 缺点：服务可用性降低                                             │
│                                                                  │
│ 适用：金融、交易等对数据一致性要求极高的场景                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ unclean.leader.election.enable = true                            │
├─────────────────────────────────────────────────────────────────┤
│ 行为：选择任意存活副本作为 Leader（可能不在 ISR 中）             │
│                                                                  │
│ 优点：最大化可用性                                               │
│ 缺点：可能丢失数据（原 Leader 未同步的消息）                     │
│                                                                  │
│ 适用：日志、监控等可容忍丢失的场景                               │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 可靠性配置矩阵

```
生产级可靠性配置（权衡可靠性与性能）：

┌───────────────────────────────────────────────────────────────────────┐
│ 配置项                          │ 高可靠推荐值   │ 高性能推荐值      │
├───────────────────────────────────────────────────────────────────────┤
│ replication.factor              │ 3             │ 2                │
│ min.insync.replicas             │ 2             │ 1                │
│ acks                            │ all           │ 1                │
│ retries                         │ MAX_INT       │ 3                │
│ enable.idempotence              │ true          │ false            │
│ unclean.leader.election.enable  │ false         │ true             │
│ log.flush.interval.messages     │ 10000         │ 100000           │
│ log.retention.hours             │ 168 (7天)     │ 24               │
└───────────────────────────────────────────────────────────────────────┘

最小 ISR 机制：
- min.insync.replicas = 2, replication.factor = 3
- 意味着必须至少有 2 个副本（1 Leader + 1 Follower）确认写入
- 如果 ISR 只剩下 Leader（1 个），acks=all 会报错
- 防止在副本不足时继续写入导致数据丢失风险
```

---

## 八、深度代码实战（Go + Java）

### 8.1 生产级 Go 消费者

```go
// mylearn/kafka-examples/consumer/consumer.go

package main

import (
    "context"
    "encoding/json"
    "fmt"
    "os"
    "os/signal"
    "sync"
    "syscall"
    "time"
    
    "github.com/IBM/sarama"
    "go.uber.org/zap"
)

// Consumer 配置
type Config struct {
    Brokers           []string      `json:"brokers"`
    Topics            []string      `json:"topics"`
    GroupID           string        `json:"group_id"`
    SessionTimeout    time.Duration `json:"session_timeout"`
    HeartbeatInterval time.Duration `json:"heartbeat_interval"`
    MaxProcessingTime time.Duration `json:"max_processing_time"`
    FetchMinBytes     int32         `json:"fetch_min_bytes"`
    FetchMaxWaitTime  time.Duration `json:"fetch_max_wait_time"`
}

// Message 处理器接口
type Handler interface {
    Handle(ctx context.Context, msg *sarama.ConsumerMessage) error
}

// ConsumerGroupHandler 实现 sarama.ConsumerGroupHandler
type ConsumerGroupHandler struct {
    handler Handler
    ready   chan bool
    logger  *zap.Logger
}

func NewConsumerGroupHandler(handler Handler, logger *zap.Logger) *ConsumerGroupHandler {
    return &ConsumerGroupHandler{
        handler: handler,
        ready:   make(chan bool),
        logger:  logger,
    }
}

func (h *ConsumerGroupHandler) Setup(sess sarama.ConsumerGroupSession) error {
    h.logger.Info("consumer group setup", 
        zap.String("member_id", sess.MemberID()),
        zap.Int("generation_id", int(sess.GenerationID())),
    )
    close(h.ready)
    return nil
}

func (h *ConsumerGroupHandler) Cleanup(sess sarama.ConsumerGroupSession) error {
    h.logger.Info("consumer group cleanup", 
        zap.String("member_id", sess.MemberID()),
    )
    return nil
}

func (h *ConsumerGroupHandler) ConsumeClaim(
    sess sarama.ConsumerGroupSession, 
    claim sarama.ConsumerGroupClaim,
) error {
    h.logger.Info("start consuming claim",
        zap.String("topic", claim.Topic()),
        zap.Int32("partition", claim.Partition()),
        zap.Int64("initial_offset", claim.InitialOffset()),
    )
    
    for {
        select {
        case msg, ok := <-claim.Messages():
            if !ok {
                h.logger.Info("message channel closed")
                return nil
            }
            
            h.logger.Debug("message received",
                zap.String("topic", msg.Topic),
                zap.Int32("partition", msg.Partition),
                zap.Int64("offset", msg.Offset),
                zap.Time("timestamp", msg.Timestamp),
            )
            
            // 带超时的处理上下文
            ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
            
            // 处理消息
            if err := h.handler.Handle(ctx, msg); err != nil {
                h.logger.Error("handle message failed",
                    zap.Error(err),
                    zap.String("topic", msg.Topic),
                    zap.Int32("partition", msg.Partition),
                    zap.Int64("offset", msg.Offset),
                )
                cancel()
                continue  // 不提交 Offset，重试
            }
            cancel()
            
            // 异步提交 Offset（高性能）
            sess.MarkMessage(msg, "")
            
        case <-sess.Context().Done():
            h.logger.Info("session context done")
            return nil
        }
    }
}

// Consumer 封装
type Consumer struct {
    client  sarama.ConsumerGroup
    handler *ConsumerGroupHandler
    wg      sync.WaitGroup
    ctx     context.Context
    cancel  context.CancelFunc
    logger  *zap.Logger
}

func NewConsumer(cfg *Config, handler Handler, logger *zap.Logger) (*Consumer, error) {
    saramaCfg := sarama.NewConfig()
    saramaCfg.Version = sarama.V3_6_0_0
    
    // 消费者组配置
    saramaCfg.Consumer.Group.Session.Timeout = cfg.SessionTimeout
    saramaCfg.Consumer.Group.Heartbeat.Interval = cfg.HeartbeatInterval
    saramaCfg.Consumer.Group.Rebalance.Timeout = cfg.SessionTimeout
    saramaCfg.Consumer.Group.Rebalance.GroupStrategies = []sarama.BalanceStrategy{
        sarama.NewBalanceStrategySticky(),  // 使用 Sticky 分配器
    }
    
    // 拉取配置
    saramaCfg.Consumer.Fetch.Min = cfg.FetchMinBytes
    saramaCfg.Consumer.Fetch.Default = 1024 * 1024  // 1MB
    saramaCfg.Consumer.Fetch.Max = 10485760  // 10MB
    saramaCfg.Consumer.MaxWaitTime = cfg.FetchMaxWaitTime
    saramaCfg.Consumer.MaxProcessingTime = cfg.MaxProcessingTime
    
    // 位移提交配置
    saramaCfg.Consumer.Offsets.AutoCommit.Enable = true
    saramaCfg.Consumer.Offsets.AutoCommit.Interval = 1 * time.Second
    saramaCfg.Consumer.Offsets.Initial = sarama.OffsetOldest  // 从最早开始
    
    // 重试配置
    saramaCfg.Consumer.Retry.Backoff = 2 * time.Second
    
    client, err := sarama.NewConsumerGroup(cfg.Brokers, cfg.GroupID, saramaCfg)
    if err != nil {
        return nil, fmt.Errorf("create consumer group: %w", err)
    }
    
    ctx, cancel := context.WithCancel(context.Background())
    
    return &Consumer{
        client:  client,
        handler: NewConsumerGroupHandler(handler, logger),
        ctx:     ctx,
        cancel:  cancel,
        logger:  logger,
    }, nil
}

func (c *Consumer) Start() {
    cfg := &Config{
        Topics: []string{"orders", "payments"},
    }
    
    c.wg.Add(1)
    go func() {
        defer c.wg.Done()
        
        for {
            select {
            case <-c.ctx.Done():
                c.logger.Info("consumer stopping")
                return
            default:
            }
            
            // 消费循环（自动处理 Rebalance）
            if err := c.client.Consume(c.ctx, cfg.Topics, c.handler); err != nil {
                c.logger.Error("consume error", zap.Error(err))
                time.Sleep(5 * time.Second)
            }
        }
    }()
}

func (c *Consumer) Stop() {
    c.cancel()
    c.wg.Wait()
    c.client.Close()
}

// ==================== 业务处理器实现 ====================

type OrderEvent struct {
    OrderID   string    `json:"order_id"`
    UserID    string    `json:"user_id"`
    Amount    float64   `json:"amount"`
    Status    string    `json:"status"`
    Timestamp time.Time `json:"timestamp"`
}

type OrderHandler struct {
    db     *Database
    cache  *Cache
    logger *zap.Logger
}

func (h *OrderHandler) Handle(ctx context.Context, msg *sarama.ConsumerMessage) error {
    var event OrderEvent
    if err := json.Unmarshal(msg.Value, &event); err != nil {
        return fmt.Errorf("unmarshal message: %w", err)
    }
    
    // 幂性检查（基于 orderID）
    if exists, _ := h.cache.Exists(ctx, event.OrderID); exists {
        h.logger.Info("duplicate event skipped", zap.String("order_id", event.OrderID))
        return nil
    }
    
    // 业务处理
    switch event.Status {
    case "created":
        return h.handleOrderCreated(ctx, &event)
    case "paid":
        return h.handleOrderPaid(ctx, &event)
    case "shipped":
        return h.handleOrderShipped(ctx, &event)
    default:
        h.logger.Warn("unknown status", zap.String("status", event.Status))
        return nil
    }
}

func (h *OrderHandler) handleOrderCreated(ctx context.Context, event *OrderEvent) error {
    // 写入数据库
    if err := h.db.CreateOrder(ctx, event); err != nil {
        return fmt.Errorf("create order: %w", err)
    }
    
    // 缓存标记已处理
    h.cache.Set(ctx, event.OrderID, "1", 24*time.Hour)
    return nil
}

// ==================== 模拟依赖 ====================
type Database struct{}
func (d *Database) CreateOrder(ctx context.Context, event *OrderEvent) error { return nil }

type Cache struct{}
func (c *Cache) Exists(ctx context.Context, key string) (bool, error) { return false, nil }
func (c *Cache) Set(ctx context.Context, key, value string, ttl time.Duration) {}

// ==================== Main ====================
func main() {
    logger, _ := zap.NewDevelopment()
    
    cfg := &Config{
        Brokers:           []string{"localhost:9092"},
        Topics:            []string{"orders"},
        GroupID:           "order-service-v1",
        SessionTimeout:    30 * time.Second,
        HeartbeatInterval: 3 * time.Second,
        MaxProcessingTime: 10 * time.Second,
        FetchMinBytes:     1,
        FetchMaxWaitTime:  500 * time.Millisecond,
    }
    
    handler := &OrderHandler{
        db:     &Database{},
        cache:  &Cache{},
        logger: logger,
    }
    
    consumer, err := NewConsumer(cfg, handler, logger)
    if err != nil {
        logger.Fatal("create consumer failed", zap.Error(err))
    }
    
    consumer.Start()
    
    // 优雅退出
    sig := make(chan os.Signal, 1)
    signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
    <-sig
    
    logger.Info("shutting down...")
    consumer.Stop()
    logger.Info("consumer stopped")
}
```

### 8.2 生产级 Java 生产者

```java
// mylearn/kafka-examples/producer/OrderProducer.java

package com.example.kafka;

import org.apache.kafka.clients.producer.*;
import org.apache.kafka.common.serialization.StringSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class OrderProducer {
    private static final Logger logger = LoggerFactory.getLogger(OrderProducer.class);
    
    private final KafkaProducer<String, String> producer;
    private final String topic;
    
    public OrderProducer(String bootstrapServers, String topic) {
        this.topic = topic;
        this.producer = createProducer(bootstrapServers);
    }
    
    private KafkaProducer<String, String> createProducer(String bootstrapServers) {
        Properties props = new Properties();
        
        // 基础配置
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.CLIENT_ID_CONFIG, "order-producer-" + System.currentTimeMillis());
        
        // ========== 可靠性配置 ==========
        // acks=all: 等待所有 ISR 副本确认
        props.put(ProducerConfig.ACKS_CONFIG, "all");
        
        // 重试次数：Integer.MAX_VALUE（配合幂等性）
        props.put(ProducerConfig.RETRIES_CONFIG, Integer.MAX_VALUE);
        
        // 重试间隔
        props.put(ProducerConfig.RETRY_BACKOFF_MS_CONFIG, 1000);
        
        // 启用幂等性（自动设置 acks=all, retries=MAX, max.in.flight=5）
        props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true);
        
        // 最大未确认请求数（幂等性下最大为 5）
        props.put(ProducerConfig.MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION, 5);
        
        // ========== 批处理与压缩 ==========
        // 批量大小：64KB
        props.put(ProducerConfig.BATCH_SIZE_CONFIG, 64 * 1024);
        
        // 批量等待时间：10ms
        props.put(ProducerConfig.LINGER_MS_CONFIG, 10);
        
        // 压缩类型：lz4（性能好）或 zstd（压缩比高）
        props.put(ProducerConfig.COMPRESSION_TYPE_CONFIG, "lz4");
        
        // 缓冲区大小：32MB
        props.put(ProducerConfig.BUFFER_MEMORY_CONFIG, 32 * 1024 * 1024L);
        
        // ========== 超时配置 ==========
        // 请求超时：30秒
        props.put(ProducerConfig.REQUEST_TIMEOUT_MS_CONFIG, 30000);
        
        // 元数据过期时间：5分钟
        props.put(ProducerConfig.METADATA_MAX_AGE_CONFIG, 300000);
        
        // ========== 事务配置（可选）==========
        // props.put(ProducerConfig.TRANSACTIONAL_ID_CONFIG, "order-tx-producer-1");
        
        return new KafkaProducer<>(props);
    }
    
    /**
     * 同步发送（可靠性优先）
     */
    public RecordMetadata sendSync(String key, String value) throws Exception {
        ProducerRecord<String, String> record = new ProducerRecord<>(topic, key, value);
        
        try {
            // 同步发送，阻塞直到收到确认
            RecordMetadata metadata = producer.send(record).get(30, TimeUnit.SECONDS);
            
            logger.info("Message sent successfully: topic={}, partition={}, offset={}, key={}",
                    metadata.topic(), metadata.partition(), metadata.offset(), key);
            
            return metadata;
        } catch (Exception e) {
            logger.error("Failed to send message: key={}, value={}", key, value, e);
            throw e;
        }
    }
    
    /**
     * 异步发送（性能优先）
     */
    public Future<RecordMetadata> sendAsync(String key, String value, SendCallback callback) {
        ProducerRecord<String, String> record = new ProducerRecord<>(topic, key, value);
        
        return producer.send(record, new Callback() {
            @Override
            public void onCompletion(RecordMetadata metadata, Exception exception) {
                if (exception != null) {
                    logger.error("Async send failed: key={}", key, exception);
                    if (callback != null) {
                        callback.onFailure(exception);
                    }
                } else {
                    logger.debug("Async send success: partition={}, offset={}",
                            metadata.partition(), metadata.offset());
                    if (callback != null) {
                        callback.onSuccess(metadata);
                    }
                }
            }
        });
    }
    
    /**
     * 带自定义分区的发送（按用户 ID 分区，保证用户内顺序）
     */
    public RecordMetadata sendToUserPartition(String userId, String value) throws Exception {
        // 使用 userId 作为 key，确保同一用户的消息进入同一分区
        return sendSync(userId, value);
    }
    
    /**
     * 事务发送（多条消息原子性）
     */
    public void sendInTransaction(java.util.List<OrderMessage> messages) throws Exception {
        // 初始化事务
        producer.initTransactions();
        
        try {
            // 开启事务
            producer.beginTransaction();
            
            for (OrderMessage msg : messages) {
                ProducerRecord<String, String> record = new ProducerRecord<>(
                        msg.getTopic(), msg.getKey(), msg.getValue());
                producer.send(record);
            }
            
            // 提交事务
            producer.commitTransaction();
            logger.info("Transaction committed successfully, {} messages", messages.size());
            
        } catch (Exception e) {
            // 回滚事务
            producer.abortTransaction();
            logger.error("Transaction aborted", e);
            throw e;
        }
    }
    
    public void close() {
        logger.info("Closing producer...");
        producer.close();
    }
    
    // ========== 回调接口 ==========
    public interface SendCallback {
        void onSuccess(RecordMetadata metadata);
        void onFailure(Exception exception);
    }
    
    // ========== 消息实体 ==========
    public static class OrderMessage {
        private String topic;
        private String key;
        private String value;
        
        // getters and setters
        public String getTopic() { return topic; }
        public void setTopic(String topic) { this.topic = topic; }
        public String getKey() { return key; }
        public void setKey(String key) { this.key = key; }
        public String getValue() { return value; }
        public void setValue(String value) { this.value = value; }
    }
}
```

### 8.3 Kafka 监控指标采集

```go
// mylearn/kafka-examples/monitoring/collector.go

package monitoring

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
    
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

// KafkaMetrics 监控指标
type KafkaMetrics struct {
    // 生产者指标
    MessagesProduced   prometheus.Counter
    ProduceErrors      prometheus.Counter
    ProduceLatency     prometheus.Histogram
    
    // 消费者指标
    MessagesConsumed   prometheus.Counter
    ConsumerLag        prometheus.GaugeVec
    ProcessErrors      prometheus.Counter
    ProcessLatency     prometheus.Histogram
    
    // Rebalance 指标
    RebalanceCounter   prometheus.Counter
    RebalanceLatency   prometheus.Histogram
}

func NewKafkaMetrics() *KafkaMetrics {
    m := &KafkaMetrics{
        MessagesProduced: prometheus.NewCounter(prometheus.CounterOpts{
            Name: "kafka_messages_produced_total",
            Help: "Total number of messages produced",
        }),
        ProduceErrors: prometheus.NewCounter(prometheus.CounterOpts{
            Name: "kafka_produce_errors_total",
            Help: "Total number of produce errors",
        }),
        ProduceLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
            Name:    "kafka_produce_latency_seconds",
            Help:    "Produce latency in seconds",
            Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
        }),
        MessagesConsumed: prometheus.NewCounter(prometheus.CounterOpts{
            Name: "kafka_messages_consumed_total",
            Help: "Total number of messages consumed",
        }),
        ConsumerLag: prometheus.NewGaugeVec(prometheus.GaugeOpts{
            Name: "kafka_consumer_lag",
            Help: "Consumer lag by topic and partition",
        }, []string{"topic", "partition", "group_id"}),
        ProcessErrors: prometheus.NewCounter(prometheus.CounterOpts{
            Name: "kafka_process_errors_total",
            Help: "Total number of message processing errors",
        }),
        ProcessLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
            Name:    "kafka_process_latency_seconds",
            Help:    "Message processing latency in seconds",
            Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
        }),
        RebalanceCounter: prometheus.NewCounter(prometheus.CounterOpts{
            Name: "kafka_rebalance_total",
            Help: "Total number of rebalances",
        }),
        RebalanceLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
            Name:    "kafka_rebalance_latency_seconds",
            Help:    "Rebalance latency in seconds",
            Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 30},
        }),
    }
    
    // 注册指标
    prometheus.MustRegister(
        m.MessagesProduced, m.ProduceErrors, m.ProduceLatency,
        m.MessagesConsumed, m.ConsumerLag, m.ProcessErrors, m.ProcessLatency,
        m.RebalanceCounter, m.RebalanceLatency,
    )
    
    return m
}

// StartMetricsServer 启动 Prometheus HTTP 服务
func (m *KafkaMetrics) StartMetricsServer(port int) {
    http.Handle("/metrics", promhttp.Handler())
    go http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
}

// 消费者延迟监控
type LagMonitor struct {
    metrics *KafkaMetrics
    admin   KafkaAdminClient
}

func (lm *LagMonitor) CollectLag(ctx context.Context, groupID string) error {
    // 获取消费者组延迟信息
    lags, err := lm.admin.ListConsumerGroupOffsets(ctx, groupID)
    if err != nil {
        return err
    }
    
    for topic, partitions := range lags {
        for partition, offset := range partitions {
            // 获取分区最新 Offset
            endOffset, err := lm.admin.GetEndOffset(ctx, topic, partition)
            if err != nil {
                continue
            }
            
            lag := endOffset - offset
            lm.metrics.ConsumerLag.WithLabelValues(
                topic, 
                fmt.Sprintf("%d", partition), 
                groupID,
            ).Set(float64(lag))
        }
    }
    return nil
}

// KafkaAdminClient 接口定义
type KafkaAdminClient interface {
    ListConsumerGroupOffsets(ctx context.Context, groupID string) (map[string]map[int32]int64, error)
    GetEndOffset(ctx context.Context, topic string, partition int32) (int64, error)
}
```

---

## 九、后端面试常见问题与答案

### 9.1 架构设计类

#### Q1: 如何设计一个支撑千万级日活的消息系统？

```
【参考答案】

1. 分区设计：
   - 计算：目标吞吐 / 单分区吞吐（约 10MB/s）
   - 示例：100MB/s 目标 → 至少 10 个分区
   - 预留 2-3 倍容量，设置 30 个分区

2. 副本设计：
   - replication.factor = 3（容忍 2 台 Broker 故障）
   - min.insync.replicas = 2（保证数据可靠性）
   - 副本跨机架/可用区部署

3. 生产者优化：
   - batch.size = 64KB, linger.ms = 10ms（批处理）
   - compression.type = lz4（压缩）
   - acks = 1（平衡可靠与性能）

4. 消费者优化：
   - 消费者数 = 分区数（最大并行度）
   - 批量消费 fetch.min.bytes = 1MB
   - 异步处理 + 批量提交 Offset

5. 监控告警：
   - 消费者延迟 > 阈值（如 1000 条）告警
   - 磁盘使用率 > 80% 告警
   - ISR 缩小告警
```

#### Q2: Kafka 为什么这么快？

```
【参考答案要点】

1. 顺序写磁盘：
   - 磁盘顺序写性能 ≈ 内存随机写（600MB/s）
   - 相比随机写（100KB/s）提升 6000 倍

2. 零拷贝技术：
   - sendfile() 系统调用
   - 数据直接从 Page Cache 发送到网卡
   - 减少 2 次 CPU 拷贝、2 次上下文切换

3. 页缓存（Page Cache）：
   - 消息先写入 Page Cache，异步刷盘
   - 消费者读消息直接从 Page Cache 读（热数据不走磁盘）

4. 批量处理：
   - 生产者批量发送（Batch）
   - 消费者批量拉取
   - 减少网络往返次数

5. 压缩：
   - 批量压缩减少网络传输和存储
   - LZ4/Snappy 压缩解压速度快

6. 分区并行：
   - 分区级别并行生产和消费
   - 水平扩展能力
```

### 9.2 核心机制类

#### Q3: Kafka 的 ISR 机制是什么？有什么作用？

```
【参考答案】

ISR（In-Sync Replicas）：与 Leader 保持同步的副本集合

工作机制：
1. Leader 维护一个 ISR 列表，包含所有"同步中"的副本
2. Follower 定期从 Leader 拉取数据
3. 如果 Follower 落后超过 replica.lag.time.max.ms（默认 30s），
   会被踢出 ISR
4. 写入时，acks=all 只等待 ISR 中的副本确认

作用：
1. 数据可靠性：只有"足够新"的副本才能参与 Leader 选举
2. 可用性权衡：允许部分慢节点不影响整体写入
3. 自动恢复：Follower 追上后可以重新加入 ISR

关键指标：
- HW（High Watermark）：ISR 中最小的 LEO，消费者只能读到 HW
- LEO（Log End Offset）：副本的日志末尾位置
```

#### Q4: Kafka 的 Rebalance 是什么？如何优化？

```
【参考答案】

Rebalance：消费者组内分区重新分配的过程

触发条件：
- 新消费者加入/退出
- 消费者心跳超时
- Topic 分区数变化

问题：
- 整个消费者组停止消费
- 分区越多，停顿越久（可达数秒）

优化方案：

1. 使用 StickyAssignor（Kafka 2.4+）：
   - 尽量保持原有分配
   - 减少不必要的分区迁移

2. 使用 Cooperative Rebalance：
   - 增量式 Rebalance
   - 只撤销需要迁移的分区，其他分区继续消费

3. 优化消费者配置：
   - session.timeout.ms = 30s（避免误判死亡）
   - heartbeat.interval.ms = 3s（保持心跳）
   - max.poll.interval.ms > 消息处理时间

4. 避免频繁 Rebalance：
   - 优雅关闭消费者（调用 close()）
   - 避免长时间 GC 停顿
   - 处理逻辑异步化，减少 poll 间隔
```

#### Q5: 如何保证消息不丢失？

```
【参考答案】

生产者端：
1. acks = all（等待所有 ISR 副本确认）
2. retries = MAX_INT（无限重试）
3. enable.idempotence = true（幂等性，防止重试重复）

Broker 端：
1. replication.factor >= 3（多副本）
2. min.insync.replicas >= 2（至少 2 个副本确认）
3. unclean.leader.election.enable = false（禁止非 ISR 副本当选 Leader）

消费者端：
1. enable.auto.commit = false（关闭自动提交）
2. 处理成功后手动提交 Offset
3. 优雅关闭时同步提交 Offset

兜底方案：
- 消息落库（生产时写入本地消息表）
- 定期对账补消息
```

### 9.3 场景应用类

#### Q6: Kafka 如何实现消息的顺序性？

```
【参考答案】

Kafka 仅保证分区内的消息顺序，不保证全局顺序。

实现方案：

1. 单分区（全局顺序）：
   - 设置 Topic 为 1 个分区
   - 所有消息进入同一分区
   - 缺点：失去并行能力，吞吐受限

2. Key 分区（业务内顺序）：
   - 使用业务 Key（如 userID、orderID）作为消息 Key
   - 相同 Key 的消息进入同一分区
   - 同一用户/订单的消息有序，不同用户并行
   
   生产者代码：
   producer.send(new ProducerRecord<>("orders", userID, message));

3. 自定义分区器：
   - 按业务维度（如地域、店铺）分区
   - 保证维度内有序，维度间并行

注意事项：
- 分区数增加后，相同 Key 可能进入不同分区
- 重试机制可能导致消息乱序（需开启幂等性）
```

#### Q7: Kafka 如何处理消息积压？

```
【参考答案】

消息积压原因分析：
1. 生产者突增流量
2. 消费者处理能力不足
3. 消费者故障/延迟

处理方案：

1. 紧急扩容消费者：
   - 增加消费者实例（不超过分区数）
   - 触发 Rebalance，分担压力

2. 优化消费速度：
   - 批量处理消息
   - 异步处理 + 批量提交 Offset
   - 增加处理协程/线程

3. 临时降级：
   - 丢弃非关键消息（如日志）
   - 采样消费

4. 增加分区：
   - 长期方案：增加 Topic 分区数
   - 需要重新分配消费者

5. 监控告警：
   - 设置消费者延迟阈值
   - 自动触发扩容

代码优化示例：
// 批量消费
for i := 0; i < batchSize; i++ {
    msg := <-consumer.Messages()
    batch = append(batch, msg)
}
processBatch(batch)  // 批量处理
commitOffset(batch)  // 批量提交
```

#### Q8: Kafka vs RabbitMQ/Redis Stream/RocketMQ 如何选型？

```
【对比表格】

┌─────────────┬────────────────┬────────────────┬────────────────┬────────────────┐
│   特性      │     Kafka      │   RabbitMQ     │  Redis Stream  │   RocketMQ     │
├─────────────┼────────────────┼────────────────┼────────────────┼────────────────┤
│ 定位        │ 流处理/大数据   │ 通用消息队列    │ 轻量级流      │ 金融级消息     │
│ 吞吐量      │ 百万级 TPS     │ 万级 TPS       │ 十万级 TPS    │ 十万级 TPS     │
│ 延迟        │ 毫秒级         │ 微秒级         │ 亚毫秒级      │ 毫秒级         │
│ 持久化      │ 磁盘顺序写     │ 内存+磁盘      │ 内存为主      │ 磁盘           │
│ 消息回溯    │ ✅ 支持        │ ❌ 消费即删    │ ✅ 支持       │ ✅ 支持        │
│ 消息查询    │ ❌ 不支持      │ ❌ 不支持      │ ❌ 不支持     │ ✅ 支持        │
│ 事务消息    │ ✅ 支持        │ ✅ 支持        │ ❌ 不支持     │ ✅ 支持        │
│ 延迟消息    │ ❌ 不支持      │ ✅ 插件支持    │ ❌ 不支持     │ ✅ 原生支持    │
│ 死信队列    │ ❌ 不支持      │ ✅ 支持        │ ❌ 不支持     │ ✅ 支持        │
│ 社区生态    │ 极丰富         │ 丰富           │ 一般          │ 国内活跃       │
│ 运维复杂度  │ 较高           │ 较低           │ 低            │ 中等           │
└─────────────┴────────────────┴────────────────┴────────────────┴────────────────┘

选型建议：
- 大数据/流处理/日志：Kafka
- 企业集成/复杂路由：RabbitMQ
- 轻量级/缓存场景：Redis Stream
- 金融交易/延迟消息：RocketMQ
```

### 9.4 原理深度类

#### Q9: Kafka 的幂等性是如何实现的？

```
【参考答案】

实现原理（PID + Sequence Number）：

1. PID（Producer ID）：
   - 生产者初始化时从 Broker 获取唯一 PID
   - 每个生产者实例全局唯一

2. Sequence Number：
   - 每个分区独立维护序列号
   - 每条消息携带 <PID, Epoch, SeqNum>

3. Broker 端去重：
   - Broker 缓存每个 <PID, Partition> 的最新 SeqNum
   - 如果收到 SeqNum <= 缓存值：重复消息，丢弃
   - 如果收到 SeqNum == 缓存值 + 1：新消息，正常写入
   - 如果收到 SeqNum > 缓存值 + 1：消息丢失，报错

限制：
1. 仅单分区单会话幂等（PID 变化则失效）
2. 需要配合事务才能实现跨分区幂等
3. 服务端需缓存 PID 状态（默认 7 天过期）
```

#### Q10: Kafka 的零拷贝是如何实现的？

```
【参考答案】

传统文件传输：
磁盘 → Page Cache → 用户缓冲区 → Socket 缓冲区 → NIC
（4 次拷贝，4 次上下文切换）

零拷贝（sendfile）：
磁盘 → Page Cache → NIC
（2 次 DMA 拷贝，2 次上下文切换）

极致零拷贝（sendfile + DMA Gather）：
磁盘 → Page Cache → NIC
（1 次 DMA 拷贝，2 次上下文切换）

Kafka 中的使用：
- 消费者拉取消息时，Broker 调用 FileChannel.transferTo()
- 数据直接从 Page Cache 发送到网卡
- 不经过 JVM 堆内存

性能对比：
- 传统方式：约 65MB/s
- 零拷贝：约 700MB/s+（10 倍提升）

为什么 Kafka 适合零拷贝：
- 消息文件是只读的（写入后不改）
- 消息格式紧凑，无序列化/反序列化开销
- 消费者通常顺序读取
```

---

## 十、性能调优与生产实践

### 10.1 Broker 调优配置

```properties
# mylearn/kafka-examples/config/server.properties

# ========== 网络与 I/O ==========
# 网络线程数（建议 = CPU 核数）
num.network.threads=8

# I/O 线程数（处理磁盘 I/O，建议 = 磁盘数 * 2）
num.io.threads=16

# Socket 缓冲区大小
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600

# ========== 日志配置 ==========
# Segment 大小（1GB，大文件减少文件句柄）
log.segment.bytes=1073741824

# 日志滚动时间（7 天）
log.roll.hours=168

# 日志保留时间（7 天）
log.retention.hours=168

# 日志保留大小（1TB）
log.retention.bytes=1099511627776

# 清理策略（delete/compact）
log.cleanup.policy=delete

# 刷新间隔（消息数）
log.flush.interval.messages=10000

# 刷新间隔（毫秒）
log.flush.interval.ms=1000

# ========== 副本配置 ==========
# 副本拉取线程数
num.replica.fetchers=4

# 副本拉取最小字节数（减少网络往返）
replica.fetch.min.bytes=1

# 副本拉取最大等待时间
replica.fetch.wait.max.ms=500

# 副本最大拉取字节数
replica.fetch.max.bytes=1048576

# 高水位检查间隔
replica.high.watermark.checkpoint.interval.ms=5000

# ========== 高可用配置 ==========
# 最小 ISR 大小（保证数据可靠性）
min.insync.replicas=2

# Leader 不均衡比例阈值
leader.imbalance.per.broker.percentage=10

# Leader 检查间隔
leader.imbalance.check.interval.seconds=300

# 是否允许非 ISR 副本当选 Leader
unclean.leader.election.enable=false

# ========== KRaft 配置 ==========
# Controller 节点列表
controller.quorum.voters=1@localhost:9093,2@localhost:9094,3@localhost:9095

# 节点角色（broker, controller, broker,controller）
process.roles=broker,controller

# Controller 监听地址
controller.listener.names=CONTROLLER
```

### 10.2 消费者延迟处理方案

```go
// mylearn/kafka-examples/consumer/lag-handler.go

package main

import (
    "context"
    "fmt"
    "sync"
    "time"
    
    "github.com/IBM/sarama"
)

// LagHandler 消费者延迟处理器
type LagHandler struct {
    consumer        sarama.ConsumerGroup
    claim           sarama.ConsumerGroupClaim
    
    // 延迟处理策略
    strategy        LagStrategy
    
    // 快速消费模式
    fastMode        bool
    fastModeTrigger int64  // 延迟超过此值触发快速模式
    
    // 采样率（快速模式下）
    sampleRate      float64
    
    // 跳过多条消息
    skipBatchSize   int
    
    logger          Logger
}

type LagStrategy int

const (
    LagStrategyNormal     LagStrategy = iota  // 正常处理
    LagStrategyFastMode                        // 快速模式（异步/批量）
    LagStrategySample                          // 采样消费
    LagStrategySkip                            // 跳过非关键消息
    LagStrategyPause                           // 暂停生产者
)

func (h *LagHandler) HandleWithLagControl(ctx context.Context) error {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case msg := <-h.claim.Messages():
            // 检查当前延迟
            lag := h.calculateLag(msg)
            
            switch h.selectStrategy(lag) {
            case LagStrategyFastMode:
                h.handleFastMode(msg)
            case LagStrategySample:
                h.handleSample(msg)
            case LagStrategySkip:
                h.handleSkip(msg)
            default:
                h.handleNormal(msg)
            }
            
        case <-ticker.C:
            // 定期上报延迟指标
            h.reportLag()
            
        case <-ctx.Done():
            return nil
        }
    }
}

func (h *LagHandler) selectStrategy(lag int64) LagStrategy {
    switch {
    case lag > 1000000:  // 超过 100 万条积压
        return LagStrategySkip
    case lag > 100000:   // 超过 10 万条
        return LagStrategySample
    case lag > 10000:    // 超过 1 万条
        return LagStrategyFastMode
    default:
        return LagStrategyNormal
    }
}

func (h *LagHandler) handleFastMode(msg *sarama.ConsumerMessage) {
    // 异步处理 + 批量提交
    h.asyncProcess(msg)
    
    // 批量提交 Offset（每 100 条）
    if msg.Offset%100 == 0 {
        h.commitBatch()
    }
}

func (h *LagHandler) handleSample(msg *sarama.ConsumerMessage) {
    // 采样处理（如只处理 1%）
    if msg.Offset%100 != 0 {
        // 跳过，直接提交 Offset
        h.claim.Messages() <- msg  // 标记已处理
        return
    }
    h.process(msg)
}

func (h *LagHandler) handleSkip(msg *sarama.ConsumerMessage) {
    // 仅处理关键消息（通过 Header 标识）
    if !h.isCritical(msg) {
        // 直接提交，跳过处理
        return
    }
    h.process(msg)
}

func (h *LagHandler) calculateLag(msg *sarama.ConsumerMessage) int64 {
    // 获取分区最新 Offset
    endOffset := h.getEndOffset(msg.Topic, msg.Partition)
    return endOffset - msg.Offset
}

// 其他辅助方法...
```

### 10.3 Docker Compose 部署配置

```yaml
# mylearn/kafka-examples/docker-compose.yml

version: '3.8'

services:
  # KRaft Controller
  controller-1:
    image: confluentinc/cp-kafka:7.5.0
    hostname: controller-1
    ports:
      - "9093:9093"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@controller-1:9093,2@controller-2:9093,3@controller-3:9093
      KAFKA_LISTENERS: CONTROLLER://0.0.0.0:9093
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LOG_DIRS: /tmp/kafka-logs
      CLUSTER_ID: kafka-cluster-1

  controller-2:
    image: confluentinc/cp-kafka:7.5.0
    hostname: controller-2
    ports:
      - "9094:9093"
    environment:
      KAFKA_NODE_ID: 2
      KAFKA_PROCESS_ROLES: controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@controller-1:9093,2@controller-2:9093,3@controller-3:9093
      KAFKA_LISTENERS: CONTROLLER://0.0.0.0:9093
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LOG_DIRS: /tmp/kafka-logs
      CLUSTER_ID: kafka-cluster-1

  controller-3:
    image: confluentinc/cp-kafka:7.5.0
    hostname: controller-3
    ports:
      - "9095:9093"
    environment:
      KAFKA_NODE_ID: 3
      KAFKA_PROCESS_ROLES: controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@controller-1:9093,2@controller-2:9093,3@controller-3:9093
      KAFKA_LISTENERS: CONTROLLER://0.0.0.0:9093
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LOG_DIRS: /tmp/kafka-logs
      CLUSTER_ID: kafka-cluster-1

  # Kafka Brokers
  broker-1:
    image: confluentinc/cp-kafka:7.5.0
    hostname: broker-1
    ports:
      - "9092:9092"
    depends_on:
      - controller-1
      - controller-2
      - controller-3
    environment:
      KAFKA_NODE_ID: 4
      KAFKA_PROCESS_ROLES: broker
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@controller-1:9093,2@controller-2:9093,3@controller-3:9093
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_LOG_DIRS: /tmp/kafka-logs
      CLUSTER_ID: kafka-cluster-1
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 2
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 2
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_MIN_INSYNC_REPLICAS: 1

  broker-2:
    image: confluentinc/cp-kafka:7.5.0
    hostname: broker-2
    ports:
      - "9096:9092"
    depends_on:
      - controller-1
      - controller-2
      - controller-3
    environment:
      KAFKA_NODE_ID: 5
      KAFKA_PROCESS_ROLES: broker
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@controller-1:9093,2@controller-2:9093,3@controller-3:9093
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9096
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_LOG_DIRS: /tmp/kafka-logs
      CLUSTER_ID: kafka-cluster-1
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 2
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 2
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_MIN_INSYNC_REPLICAS: 1

  # Kafka UI
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: broker-1:9092,broker-2:9092
    depends_on:
      - broker-1
      - broker-2
```

---

## 总结

本文档从 JD 要求出发，深入解析了 Kafka 的核心机制：

1. **架构层面**：理解 KRaft 元数据管理、ISR 副本机制、Rebalance 协议
2. **存储层面**：掌握 Segment 分段存储、稀疏索引、零拷贝技术
3. **生产消费**：熟悉幂等性、事务消息、位移提交策略
4. **面试准备**：掌握常见面试问题的回答思路

后续学习建议：
- 阅读 Kafka 官方文档和 KIP（Kafka Improvement Proposals）
- 实践搭建 Kafka 集群，压测验证配置效果
- 学习 Kafka Streams / Kafka Connect 构建数据 Pipeline

---

**参考资源**：
- [Kafka 官方文档](https://kafka.apache.org/documentation/)
- 《Kafka 权威指南》（第2版）
- 《数据密集型应用系统设计》第 3 章
- [Confluent 博客](https://www.confluent.io/blog/)
