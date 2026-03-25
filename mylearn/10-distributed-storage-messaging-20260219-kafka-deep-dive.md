# Kafka 深度解析：从架构原理到生产实战

**日期**: 2026-02-19
**学习路径**: 10 - 分布式存储与消息系统
**对话主题**: Kafka 核心架构、存储机制、消息语义、高可用、性能调优与 Go 实战

## 问题背景

JD 要求：
- **领域二**："负责数据采集、清洗、去重与质量评估系统的设计与开发"
- **领域二**："持续优化数据处理各环节的性能与吞吐，确保数据管道的稳定高效"
- **领域一**："负责核心服务的性能优化、分布式系统可靠性保障"
- **核心要求**："对分布式系统有深刻理解与实践经验，能够设计高可用、高可靠的系统架构"

Kafka 是大规模数据 Pipeline 的核心组件，需要达到**精通级别**。

---

## 一、Kafka 整体架构

### 1.1 核心组件关系

```
┌─────────────────────────────────────────────────────────────┐
│                      Kafka Cluster                          │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Broker 0 │  │ Broker 1 │  │ Broker 2 │                 │
│  │          │  │          │  │          │                 │
│  │ P0(L)    │  │ P0(F)    │  │ P0(F)    │  ← Topic A     │
│  │ P1(F)    │  │ P1(L)    │  │ P1(F)    │                 │
│  │ P2(F)    │  │ P2(F)    │  │ P2(L)    │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
│       ↑              ↑              ↑                       │
│       └──────────────┼──────────────┘                       │
│              Controller (KRaft)                             │
└─────────────────────────────────────────────────────────────┘
        ↑                                    ↑
   ┌─────────┐                         ┌──────────┐
   │Producer │                         │Consumer  │
   │  Group  │                         │  Group   │
   └─────────┘                         └──────────┘
```

**关键概念**：
- **Broker**：Kafka 服务节点，每个 Broker 存储多个 Partition 的数据
- **Topic**：逻辑消息分类，一个 Topic 包含多个 Partition
- **Partition**：物理存储单元，是 Kafka 并行度的基本单位
- **Segment**：Partition 内的日志段文件，是实际磁盘文件
- **Replica**：副本，分为 Leader 和 Follower
- **ISR (In-Sync Replicas)**：与 Leader 保持同步的副本集合
- **Controller**：集群控制器，负责分区 Leader 选举和元数据管理

### 1.2 为什么 Kafka 这么快？—— 四大核心设计

```
性能关键设计：
┌─────────────────────────────────────────────┐
│ 1. 顺序写磁盘（Sequential I/O）              │
│    - 追加写入，不随机修改                      │
│    - 顺序写磁盘速度 ≈ 600MB/s（SSD更快）      │
│    - 随机写磁盘速度 ≈ 100KB/s                 │
│    - 差距约 6000 倍                           │
├─────────────────────────────────────────────┤
│ 2. 零拷贝（Zero-Copy）sendfile()             │
│    传统路径: 磁盘→内核缓冲→用户缓冲→Socket缓冲→NIC │
│    零拷贝:   磁盘→内核缓冲→NIC（DMA直传）       │
│    减少 2 次 CPU 拷贝 + 2 次上下文切换          │
├─────────────────────────────────────────────┤
│ 3. 页缓存（Page Cache）                      │
│    - 利用 OS 页缓存，不自己管理内存             │
│    - 写入时先写页缓存，OS 异步刷盘              │
│    - 读取时命中页缓存直接返回                   │
│    - GC 友好：数据不在 JVM 堆内                │
├─────────────────────────────────────────────┤
│ 4. 批处理 + 压缩                             │
│    - Producer 端批量发送（batch.size + linger.ms）│
│    - 支持 gzip/snappy/lz4/zstd 压缩          │
│    - 网络传输量大幅减少                        │
└─────────────────────────────────────────────┘
```

### 1.3 零拷贝机制详解

```
传统数据传输（4次拷贝 + 4次上下文切换）：
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────┐
│ Disk │───→│ Kernel   │───→│ User     │───→│ Socket   │───→│ NIC │
│      │DMA │ Buffer   │CPU │ Buffer   │CPU │ Buffer   │DMA │     │
└──────┘    └──────────┘    └──────────┘    └──────────┘    └─────┘
  read()系统调用(切换)    拷贝到用户空间    write()系统调用(切换)

零拷贝 sendfile()（2次拷贝 + 2次上下文切换）：
┌──────┐    ┌──────────┐                                   ┌─────┐
│ Disk │───→│ Kernel   │──────────────────────────────────→│ NIC │
│      │DMA │ Buffer   │         DMA Scatter/Gather        │     │
└──────┘    └──────────┘                                   └─────┘
  sendfile()系统调用(切换)    数据不经过用户空间
```

**Linux 源码层面**：Kafka 的 `FileChannel.transferTo()` 底层调用 `sendfile()` 系统调用，
在 Linux 2.4+ 内核中，配合 DMA Scatter/Gather，数据从磁盘到网卡全程不经过 CPU 拷贝。

---

## 二、存储机制深度解析

### 2.1 日志段（Segment）结构

```
Topic: orders, Partition: 0
目录: /kafka-logs/orders-0/

├── 00000000000000000000.log        ← 消息数据文件（第一个 Segment）
├── 00000000000000000000.index      ← 偏移量索引（稀疏索引）
├── 00000000000000000000.timeindex  ← 时间戳索引
├── 00000000000005367851.log        ← 第二个 Segment（baseOffset=5367851）
├── 00000000000005367851.index
├── 00000000000005367851.timeindex
├── 00000000000009834567.log        ← 第三个 Segment
├── 00000000000009834567.index
├── 00000000000009834567.timeindex
└── leader-epoch-checkpoint          ← Leader Epoch 信息
```

**文件命名规则**：文件名是该 Segment 中第一条消息的 offset，20位数字左补零。

### 2.2 消息存储格式（RecordBatch，Kafka 2.0+）

```
RecordBatch 结构（磁盘上的二进制格式）：
┌─────────────────────────────────────────────────┐
│ baseOffset: int64          (8 bytes)            │ ← 批次第一条消息的 offset
│ batchLength: int32         (4 bytes)            │ ← 批次总字节数
│ partitionLeaderEpoch: int32(4 bytes)            │
│ magic: int8                (1 byte)  = 2        │ ← 版本号
│ crc: uint32                (4 bytes)            │ ← CRC32C 校验
│ attributes: int16          (2 bytes)            │ ← 压缩/事务/时间戳类型
│ lastOffsetDelta: int32     (4 bytes)            │
│ baseTimestamp: int64       (8 bytes)            │
│ maxTimestamp: int64        (8 bytes)            │
│ producerId: int64          (8 bytes)            │ ← 幂等/事务用
│ producerEpoch: int16       (2 bytes)            │
│ baseSequence: int32        (4 bytes)            │ ← 幂等用序列号
│ records count: int32       (4 bytes)            │
│ ┌─────────────────────────────────────────────┐ │
│ │ Record 0                                    │ │
│ │  length: varint                             │ │
│ │  attributes: int8                           │ │
│ │  timestampDelta: varlong                    │ │
│ │  offsetDelta: varint                        │ │
│ │  keyLength: varint + key bytes              │ │
│ │  valueLength: varint + value bytes          │ │
│ │  headers: [Header...]                       │ │
│ ├─────────────────────────────────────────────┤ │
│ │ Record 1 ...                                │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**关键设计点**：
- 使用 **varint** 编码减少小数字的存储空间
- **delta 编码**：timestamp 和 offset 存储相对于 base 的差值
- **批次级压缩**：整个 RecordBatch 的 records 部分一起压缩
- **CRC32C**：硬件加速的校验算法，比 CRC32 快 8 倍

### 2.3 稀疏索引机制

```
.index 文件结构（偏移量索引）：
┌──────────────────────────────────┐
│ relativeOffset(4B) | position(4B)│  ← 每条索引 8 字节
├──────────────────────────────────┤
│     0              |     0       │  ← offset=baseOffset+0, 文件位置 0
│   512              |  16384      │  ← offset=baseOffset+512, 位置 16384
│  1024              |  32890      │  ← offset=baseOffset+1024, 位置 32890
│  1536              |  49152      │
│  ...               |  ...        │
└──────────────────────────────────┘

查找 offset=baseOffset+700 的过程：
1. 二分查找 .index → 找到 [512, 16384]（最近的小于等于 700 的条目）
2. 从 .log 文件位置 16384 开始顺序扫描
3. 逐条读取 Record 直到找到 offset=700

.timeindex 文件结构（时间戳索引）：
┌──────────────────────────────────────────┐
│ timestamp(8B) | relativeOffset(4B)       │  ← 每条 12 字节
├──────────────────────────────────────────┤
│ 1708300000000 |     0                    │
│ 1708300010000 |   512                    │
│ 1708300020000 |  1024                    │
└──────────────────────────────────────────┘
```

**索引间隔**：由 `log.index.interval.bytes` 控制（默认 4096 字节），
即每写入 4KB 消息数据就在索引中添加一条记录。这是空间与查找速度的权衡。

### 2.4 日志清理策略

```
两种清理策略：

1. Delete（删除）—— log.cleanup.policy=delete
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Segment 1│ │Segment 2│ │Segment 3│ │Segment 4│
   │ 已过期  │ │ 已过期  │ │  活跃   │ │  活跃   │
   └────┬────┘ └────┬────┘ └─────────┘ └─────────┘
        ↓           ↓
      删除         删除

   触发条件：
   - log.retention.hours=168（默认7天）
   - log.retention.bytes=-1（默认不限制）
   - log.segment.bytes=1073741824（1GB 切分新 Segment）

2. Compact（压缩）—— log.cleanup.policy=compact
   压缩前：                    压缩后：
   K1:V1 → K2:V1 → K1:V2     K2:V1 → K1:V3 → K3:V1
   → K3:V1 → K1:V3            （每个 Key 只保留最新值）

   适用场景：
   - __consumer_offsets（消费者位移）
   - Kafka Streams 状态存储
   - CDC 场景（保留每行最新状态）
```

---

## 三、生产者（Producer）核心机制

### 3.1 消息发送流程

```
Producer 内部架构：
┌─────────────────────────────────────────────────────────┐
│                     Producer                             │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ 拦截器    │───→│ 序列化器      │───→│ 分区器        │  │
│  │Interceptor│    │ Serializer   │    │ Partitioner   │  │
│  └──────────┘    └──────────────┘    └───────┬───────┘  │
│                                              │          │
│                                              ↓          │
│  ┌──────────────────────────────────────────────────┐   │
│  │           RecordAccumulator（消息累加器）          │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │   │
│  │  │TP0: Deque   │ │TP1: Deque   │ │TP2: Deque  │ │   │
│  │  │[Batch][Batch]│ │[Batch]      │ │[Batch]     │ │   │
│  │  └─────────────┘ └─────────────┘ └────────────┘ │   │
│  └──────────────────────────┬───────────────────────┘   │
│                             │                            │
│                             ↓                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Sender 线程（独立线程）               │   │
│  │  - 将 Batch 按 Broker 分组                        │   │
│  │  - 构建 ProduceRequest                            │   │
│  │  - 管理 InFlightRequests（max.in.flight=5）       │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 分区策略

```go
// Go 中使用 sarama 库的分区策略示例
// 默认分区策略逻辑（伪代码）：
func partition(key []byte, numPartitions int32) int32 {
    if key == nil {
        // 无 Key：Sticky Partitioner（Kafka 2.4+）
        // 同一批次的消息发往同一分区，批次满后切换
        // 旧版本是 Round-Robin
        return stickyPartition(numPartitions)
    }
    // 有 Key：murmur2 哈希
    hash := murmur2(key)
    return int32(hash) % numPartitions  // 注意：不取绝对值
}
```

**Sticky Partitioner 的优势**：
- 旧版 Round-Robin：每条消息分到不同分区 → 每个 Batch 只有 1 条消息 → 小包多
- Sticky：同批消息发往同一分区 → Batch 更满 → 减少请求数 → 吞吐提升 50%+

### 3.3 幂等性（Idempotent Producer）

```
问题：网络抖动导致 Producer 重试，Broker 可能收到重复消息

解决方案：enable.idempotence=true

┌──────────┐                    ┌──────────┐
│ Producer │                    │  Broker  │
│ PID=1000 │                    │          │
│ Seq=0    │───── Msg(Seq=0) ──→│ 接收 ✓   │
│          │←──── ACK ─────────│          │
│ Seq=1    │───── Msg(Seq=1) ──→│ 接收 ✓   │
│          │←──── ACK(丢失) ───│          │
│ Seq=1    │───── Msg(Seq=1) ──→│ 重复！   │
│          │     （重试）        │ Seq=1已有│
│          │←── DuplicateSeq ──│ 丢弃 ✓   │
└──────────┘                    └──────────┘

核心机制：
- PID (Producer ID)：每个 Producer 实例唯一标识
- Sequence Number：每个 <PID, Partition> 维护单调递增序列号
- Broker 端维护 Map<PID, Map<Partition, LastSeq>>
- 如果收到的 Seq <= LastSeq，判定为重复，返回 DuplicateSequenceException
- 如果收到的 Seq > LastSeq + 1，判定为乱序，返回 OutOfOrderSequenceException

限制：
- 幂等性只保证单分区内的 Exactly Once
- 跨分区需要事务（Transactional Producer）
- max.in.flight.requests.per.connection <= 5（Kafka 2.0+ 自动设置）
```

### 3.4 事务消息（Transactional Producer）

```
场景：消费 Topic A → 处理 → 写入 Topic B + 提交 offset，要求原子性

事务流程：
┌──────────┐         ┌────────────────┐         ┌──────────┐
│ Producer │         │ Transaction    │         │  Broker  │
│          │         │ Coordinator    │         │          │
│          │         │ (某个Broker)    │         │          │
└────┬─────┘         └───────┬────────┘         └────┬─────┘
     │                       │                       │
     │ 1.FindCoordinator     │                       │
     │──────────────────────→│                       │
     │                       │                       │
     │ 2.InitProducerId      │                       │
     │──────────────────────→│                       │
     │   (获取PID+Epoch)     │                       │
     │←──────────────────────│                       │
     │                       │                       │
     │ 3.BeginTransaction    │                       │
     │  (本地标记)            │                       │
     │                       │                       │
     │ 4.AddPartitionsToTxn  │                       │
     │──────────────────────→│                       │
     │                       │                       │
     │ 5.Produce(带PID+Epoch)│                       │
     │──────────────────────────────────────────────→│
     │                       │                       │
     │ 6.AddOffsetsToTxn     │                       │
     │──────────────────────→│                       │
     │                       │                       │
     │ 7.TxnOffsetCommit     │                       │
     │──────────────────────────────────────────────→│
     │                       │                       │
     │ 8.EndTxn(COMMIT)      │                       │
     │──────────────────────→│                       │
     │                       │ 9.WriteTxnMarker      │
     │                       │──────────────────────→│
     │                       │  (写入 COMMIT 标记)    │
     │                       │                       │

__transaction_state 内部 Topic：
- 存储事务状态（Ongoing/PrepareCommit/CompleteCommit/PrepareAbort/CompleteAbort）
- 50 个分区，transactional.id 哈希决定分区
- Transaction Coordinator 是该分区的 Leader Broker
```

### 3.5 关键 Producer 配置深度解读

```yaml
# ===== 可靠性配置 =====
acks: "all"                    # -1/all: 等待 ISR 所有副本确认
                               # 1: 只等 Leader 确认（默认）
                               # 0: 不等确认（最快但可能丢消息）

retries: 2147483647            # 重试次数（幂等模式下自动设为 MAX_INT）
delivery.timeout.ms: 120000   # 消息发送总超时（含重试），超过则放弃

enable.idempotence: true       # 开启幂等性
transactional.id: "my-txn-id"  # 开启事务（设置后自动开启幂等）

# ===== 性能配置 =====
batch.size: 16384              # 单个 Batch 最大字节数（默认 16KB）
                               # 生产环境建议 32KB-64KB
linger.ms: 0                   # 发送延迟（默认 0 = 立即发送）
                               # 生产环境建议 5-10ms，等待更多消息凑批
buffer.memory: 33554432        # Producer 缓冲区总大小（默认 32MB）
max.block.ms: 60000            # 缓冲区满时 send() 阻塞的最大时间

compression.type: "lz4"        # 压缩算法
                               # none: 不压缩
                               # gzip: 压缩率最高，CPU 消耗大
                               # snappy: 均衡选择
                               # lz4: 速度最快，推荐生产使用
                               # zstd: 压缩率接近 gzip，速度接近 lz4

max.in.flight.requests.per.connection: 5
                               # 单连接最大未确认请求数
                               # 幂等模式下 <=5 保证顺序
                               # 非幂等模式下 =1 才能保证顺序

# ===== 压缩效果对比（100万条 1KB 消息）=====
# 算法      压缩率    压缩速度    解压速度
# none      1.0x      -          -
# gzip      ~2.5x     ~50MB/s    ~250MB/s
# snappy    ~1.7x     ~250MB/s   ~500MB/s
# lz4       ~2.1x     ~400MB/s   ~800MB/s
# zstd      ~2.4x     ~200MB/s   ~600MB/s
```

---

## 四、消费者（Consumer）核心机制

### 4.1 消费者组（Consumer Group）

```
Consumer Group 分区分配：

Topic: orders (6 个分区)

场景1：3个消费者
┌──────────┐  ┌──────────┐  ┌──────────┐
│Consumer 0│  │Consumer 1│  │Consumer 2│
│ P0, P1   │  │ P2, P3   │  │ P4, P5   │
└──────────┘  └──────────┘  └──────────┘

场景2：6个消费者（最佳并行度）
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
│ C0 │ │ C1 │ │ C2 │ │ C3 │ │ C4 │ │ C5 │
│ P0 │ │ P1 │ │ P2 │ │ P3 │ │ P4 │ │ P5 │
└────┘ └────┘ └────┘ └────┘ └────┘ └────┘

场景3：8个消费者（2个空闲！）
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
│ C0 │ │ C1 │ │ C2 │ │ C3 │ │ C4 │ │ C5 │ │ C6 │ │ C7 │
│ P0 │ │ P1 │ │ P2 │ │ P3 │ │ P4 │ │ P5 │ │空闲│ │空闲│
└────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘

核心规则：
- 一个分区只能被同一消费者组中的一个消费者消费
- 消费者数量 > 分区数时，多余的消费者空闲
- 不同消费者组之间互不影响（广播模式）
```

### 4.2 再平衡（Rebalance）协议

```
Rebalance 触发条件：
1. 消费者加入/离开消费者组
2. 消费者崩溃（session.timeout.ms 超时未发心跳）
3. 消费者处理超时（max.poll.interval.ms 超时未 poll）
4. Topic 分区数变化
5. 订阅的 Topic 列表变化

Rebalance 过程（Eager 协议 → 全量重分配，有 STW）：
┌──────────┐  ┌──────────┐  ┌──────────────┐
│Consumer 0│  │Consumer 1│  │Group         │
│          │  │(新加入)   │  │Coordinator   │
└────┬─────┘  └────┬─────┘  └──────┬───────┘
     │              │               │
     │  JoinGroup   │               │
     │─────────────────────────────→│
     │              │  JoinGroup    │
     │              │──────────────→│
     │              │               │
     │  JoinResponse(Leader=C0)     │  ← C0 被选为 Group Leader
     │←─────────────────────────────│
     │              │  JoinResponse │
     │              │←──────────────│
     │              │               │
     │  C0 执行分区分配算法          │
     │  SyncGroup(分配结果)          │
     │─────────────────────────────→│
     │              │  SyncGroup    │
     │              │──────────────→│
     │              │               │
     │  SyncResponse(你的分区)       │
     │←─────────────────────────────│
     │              │  SyncResponse │
     │              │←──────────────│

Cooperative Sticky（增量再平衡，Kafka 2.4+）：
- 不再全量 revoke 所有分区
- 只 revoke 需要迁移的分区
- 大幅减少 STW 时间
- 分两轮完成：第一轮 revoke，第二轮 assign
```

### 4.3 分区分配策略

```
三种内置策略：

1. RangeAssignor（默认）
   Topic T1(P0,P1,P2), T2(P0,P1,P2), Consumers: C0, C1

   T1: C0=[P0,P1], C1=[P2]      ← 按 Topic 独立分配
   T2: C0=[P0,P1], C1=[P2]      ← C0 总是多分一个
   问题：多 Topic 时 C0 负载偏重

2. RoundRobinAssignor
   所有分区排序后轮询分配：
   C0=[T1P0, T1P2, T2P1]
   C1=[T1P1, T2P0, T2P2]        ← 更均匀

3. StickyAssignor（推荐）
   - 尽量均匀分配
   - Rebalance 时尽量保持原有分配不变
   - 减少分区迁移，降低 Rebalance 开销

4. CooperativeStickyAssignor（Kafka 2.4+，最推荐）
   - 基于 Sticky 策略
   - 支持增量 Rebalance
   - 不需要全量 revoke
```

### 4.4 Offset 管理

```
Offset 存储位置：__consumer_offsets（内部 Topic，50 个分区）

Key: <group.id, topic, partition>
Value: <offset, metadata, timestamp>

提交方式：
┌─────────────────────────────────────────────────────────┐
│ 1. 自动提交（enable.auto.commit=true）                   │
│    - 每 auto.commit.interval.ms（默认5s）自动提交        │
│    - 风险：消费后未处理完就提交 → 消息丢失                 │
│    - 风险：处理完未提交就崩溃 → 重复消费                   │
│                                                          │
│ 2. 同步手动提交（commitSync）                             │
│    - 阻塞直到提交成功                                     │
│    - 可靠但影响吞吐                                       │
│                                                          │
│ 3. 异步手动提交（commitAsync）                            │
│    - 非阻塞，通过回调处理结果                              │
│    - 高吞吐但可能提交失败                                  │
│                                                          │
│ 4. 最佳实践：异步提交 + 关闭时同步提交                     │
│    正常消费 → commitAsync()                               │
│    Consumer 关闭 → commitSync()（确保最后一次提交成功）     │
└─────────────────────────────────────────────────────────┘

Offset 重置策略（auto.offset.reset）：
- earliest: 从最早的消息开始消费
- latest: 从最新的消息开始消费（默认）
- none: 没有已提交 offset 时抛异常
```

---

## 五、高可用与容错机制

### 5.1 副本机制与 ISR

```
副本状态模型：
┌─────────────────────────────────────────────────┐
│              AR (Assigned Replicas)              │
│  ┌───────────────────────────────────────────┐  │
│  │           ISR (In-Sync Replicas)          │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│  │  │ Leader   │ │Follower 1│ │Follower 2│  │  │
│  │  │ Broker 0 │ │ Broker 1 │ │ Broker 2 │  │  │
│  │  └──────────┘ └──────────┘ └──────────┘  │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  OSR (Out-of-Sync Replicas) = AR - ISR          │
│  ┌──────────┐                                    │
│  │Follower 3│  ← 落后太多，被踢出 ISR            │
│  │ Broker 3 │                                    │
│  └──────────┘                                    │
└─────────────────────────────────────────────────┘

ISR 收缩条件：
- replica.lag.time.max.ms=30000（默认30s）
  Follower 超过 30s 没有向 Leader 发送 Fetch 请求
  或 Fetch 请求的 offset 落后 Leader 太多

ISR 扩张条件：
- Follower 追上 Leader 的 LEO（Log End Offset）
```

### 5.2 HW（High Watermark）与 LEO

```
HW 与 LEO 的关系：

Leader (Broker 0):
  消息: [M0][M1][M2][M3][M4][M5][M6]
                              ↑      ↑
                              HW     LEO
Follower 1 (Broker 1):
  消息: [M0][M1][M2][M3][M4][M5]
                              ↑    ↑
                              HW   LEO
Follower 2 (Broker 2):
  消息: [M0][M1][M2][M3][M4]
                           ↑  ↑
                           HW LEO

HW = min(所有 ISR 副本的 LEO) = 5
消费者只能看到 HW 之前的消息（M0-M4）

问题：HW 更新有延迟，可能导致数据不一致
解决：Leader Epoch（Kafka 0.11+）
- 每次 Leader 切换，Epoch +1
- Follower 恢复时先查询 Leader Epoch 对应的 offset
- 避免基于 HW 截断导致的数据丢失
```

### 5.3 Leader 选举

```
Leader 选举场景：

1. 正常选举（ISR 中选）：
   ISR = [Broker0(Leader), Broker1, Broker2]
   Broker0 宕机 → Controller 从 ISR 中选择 Broker1 为新 Leader
   ISR = [Broker1(Leader), Broker2]

2. Unclean Leader Election（非 ISR 选举）：
   unclean.leader.election.enable=false（默认，推荐）
   ISR 全部宕机 → 分区不可用，等待 ISR 恢复

   unclean.leader.election.enable=true
   ISR 全部宕机 → 从 OSR 中选 Leader → 可能丢数据！

3. Preferred Leader Election：
   preferred.leader = AR 列表中的第一个 Broker
   auto.leader.rebalance.enable=true（默认）
   定期检查并将 Leader 迁回 preferred leader，保持负载均衡

Controller 选举（KRaft 模式前）：
- 所有 Broker 竞争在 ZooKeeper 创建 /controller 临时节点
- 创建成功的成为 Controller
- Controller 负责：分区 Leader 选举、ISR 变更、Broker 上下线
```

---

## 六、KRaft 模式（Kafka 3.0+）

### 6.1 为什么要去 ZooKeeper？

```
ZooKeeper 模式的问题：
┌─────────────────────────────────────────────────┐
│ 1. 运维复杂：需要独立维护 ZK 集群（3-5节点）      │
│ 2. 元数据瓶颈：大集群（>100K分区）ZK 成为瓶颈     │
│ 3. Controller 故障恢复慢：需要从 ZK 全量加载元数据 │
│ 4. 脑裂风险：ZK session 超时可能导致双 Controller  │
│ 5. 监控困难：需要同时监控 Kafka + ZK 两套系统      │
└─────────────────────────────────────────────────┘

KRaft 模式的优势：
┌─────────────────────────────────────────────────┐
│ 1. 架构简化：不再依赖 ZooKeeper                   │
│ 2. 元数据性能：基于 Raft 的内部元数据日志           │
│ 3. 快速恢复：Controller 故障切换 < 10ms           │
│ 4. 扩展性：支持百万级分区                          │
│ 5. 统一运维：只需管理 Kafka 集群                   │
└─────────────────────────────────────────────────┘
```

### 6.2 KRaft 架构

```
KRaft 集群角色：
┌──────────────────────────────────────────────────────┐
│                    KRaft Cluster                      │
│                                                       │
│  Controller 节点（Raft Quorum）：                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐       │
│  │Controller 0│ │Controller 1│ │Controller 2│       │
│  │  (Leader)  │ │ (Follower) │ │ (Follower) │       │
│  │            │ │            │ │            │       │
│  │ __cluster  │ │ __cluster  │ │ __cluster  │       │
│  │ _metadata  │ │ _metadata  │ │ _metadata  │       │
│  └────────────┘ └────────────┘ └────────────┘       │
│        ↕               ↕               ↕             │
│  Broker 节点：                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Broker 0 │ │ Broker 1 │ │ Broker 2 │            │
│  │ 数据存储  │ │ 数据存储  │ │ 数据存储  │            │
│  └──────────┘ └──────────┘ └──────────┘            │
└──────────────────────────────────────────────────────┘

__cluster_metadata Topic：
- 存储所有集群元数据（Topic、分区、副本、ACL等）
- 使用 Raft 协议保证一致性
- 所有 Broker 通过 Fetch 请求获取元数据更新
- 类似于 etcd 的 WAL 日志

混合模式（process.roles=broker,controller）：
- 小集群可以让同一节点同时担任 Broker 和 Controller
- 生产环境建议分离部署
```

---

## 七、消息投递语义详解

### 7.1 三种语义对比

```
┌──────────────┬──────────────────────────────────────────┐
│    语义       │              实现方式                     │
├──────────────┼──────────────────────────────────────────┤
│ At Most Once │ Producer: acks=0 或 acks=1 + 不重试      │
│ （最多一次）  │ Consumer: 先提交 offset，再处理消息       │
│              │ 风险：消息可能丢失                         │
│              │ 场景：日志收集、监控指标（允许少量丢失）     │
├──────────────┼──────────────────────────────────────────┤
│ At Least Once│ Producer: acks=all + retries>0            │
│ （至少一次）  │ Consumer: 先处理消息，再提交 offset       │
│              │ 风险：消息可能重复                         │
│              │ 场景：大多数业务场景（配合幂等处理）        │
├──────────────┼──────────────────────────────────────────┤
│ Exactly Once │ Producer: 幂等 + 事务                     │
│ （精确一次）  │ Consumer: read_committed + 事务           │
│              │ 开销：性能下降 ~20%                        │
│              │ 场景：金融交易、计费系统                    │
└──────────────┴──────────────────────────────────────────┘
```

### 7.2 Exactly Once 端到端实现

```
Consume-Transform-Produce 模式：

┌──────────┐     ┌──────────────────┐     ┌──────────┐
│ Source   │     │   Application    │     │  Sink    │
│ Topic    │────→│                  │────→│  Topic   │
│          │     │ 1. consume()     │     │          │
│          │     │ 2. transform()   │     │          │
│          │     │ 3. produce()     │     │          │
│          │     │ 4. commitOffset()│     │          │
└──────────┘     └──────────────────┘     └──────────┘

事务保证：步骤 3 和 4 在同一个事务中原子完成

Consumer 端配置：
  isolation.level=read_committed
  - 只读取已提交事务的消息
  - 未提交/已中止事务的消息被过滤

  isolation.level=read_uncommitted（默认）
  - 读取所有消息，包括未提交事务的
```

---

## 八、Kafka Streams 与 Kafka Connect

### 8.1 Kafka Connect

```
Kafka Connect 架构：
┌─────────────────────────────────────────────────────┐
│                  Kafka Connect Cluster               │
│                                                      │
│  ┌──────────────────┐    ┌──────────────────┐       │
│  │  Source Connector │    │  Sink Connector  │       │
│  │                  │    │                  │       │
│  │  MySQL(Debezium) │    │  Elasticsearch   │       │
│  │  PostgreSQL      │    │  HDFS            │       │
│  │  MongoDB         │    │  S3              │       │
│  │  File            │    │  JDBC            │       │
│  └────────┬─────────┘    └────────┬─────────┘       │
│           │                       │                  │
│           ↓                       ↑                  │
│  ┌──────────────────────────────────────────┐       │
│  │              Kafka Cluster               │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘

核心概念：
- Connector：定义数据源/目标的连接配置
- Task：实际执行数据传输的工作单元
- Worker：运行 Task 的 JVM 进程
- Converter：数据格式转换（JSON、Avro、Protobuf）
- Transform（SMT）：单条消息转换（过滤、路由、字段操作）

CDC 场景（Debezium + Kafka Connect）：
MySQL Binlog → Debezium Source Connector → Kafka Topic
→ Elasticsearch Sink Connector → ES Index
```

### 8.2 Kafka Streams

```
Kafka Streams 核心抽象：

KStream（事件流）：每条记录是独立事件
  [("key1","v1"), ("key2","v2"), ("key1","v3")]
  → 三条独立记录

KTable（变更日志）：每个 Key 只保留最新值
  [("key1","v1"), ("key2","v2"), ("key1","v3")]
  → key1=v3, key2=v2

GlobalKTable：全量广播到每个实例

核心操作：
- filter / map / flatMap / groupBy
- join（KStream-KStream, KStream-KTable, KTable-KTable）
- aggregate / reduce / count
- windowed operations（Tumbling, Hopping, Sliding, Session）

状态存储：
- RocksDB（默认本地状态存储）
- Changelog Topic（状态备份，用于故障恢复）
- Interactive Queries（查询本地状态）
```

---

## 九、性能调优实战

### 9.1 Broker 端调优

```yaml
# ===== 网络与线程 =====
num.network.threads: 8              # 网络线程数（处理网络请求）
num.io.threads: 16                  # IO 线程数（处理磁盘读写）
socket.send.buffer.bytes: 1048576   # Socket 发送缓冲区 1MB
socket.receive.buffer.bytes: 1048576 # Socket 接收缓冲区 1MB
socket.request.max.bytes: 104857600 # 单个请求最大 100MB

# ===== 日志配置 =====
log.segment.bytes: 1073741824       # Segment 大小 1GB
log.retention.hours: 168            # 保留 7 天
log.retention.bytes: -1             # 不限制总大小
log.cleanup.policy: delete          # 清理策略
log.index.interval.bytes: 4096     # 索引间隔 4KB

# ===== 副本配置 =====
num.replica.fetchers: 4             # 副本拉取线程数
replica.fetch.max.bytes: 10485760   # 副本拉取最大 10MB
replica.lag.time.max.ms: 30000      # ISR 超时 30s

# ===== 性能关键 =====
message.max.bytes: 10485760         # 单条消息最大 10MB
num.partitions: 6                   # 默认分区数
default.replication.factor: 3       # 默认副本数
min.insync.replicas: 2              # 最小同步副本数

# ===== OS 层面优化 =====
# vm.swappiness=1                   # 几乎禁用 swap
# vm.dirty_ratio=60                 # 脏页比例
# vm.dirty_background_ratio=5       # 后台刷盘比例
# net.core.wmem_default=131072      # Socket 写缓冲
# net.core.rmem_default=131072      # Socket 读缓冲
# 文件系统推荐 XFS（比 ext4 更适合 Kafka 的顺序写）
```

### 9.2 分区数规划

```
分区数计算公式：

目标吞吐量 = T（MB/s）
单分区 Producer 吞吐 = Pp（MB/s）
单分区 Consumer 吞吐 = Cp（MB/s）

分区数 = max(T/Pp, T/Cp)

示例：
- 目标吞吐：1000 MB/s
- 单分区 Producer 吞吐：100 MB/s
- 单分区 Consumer 吞吐：50 MB/s
- 分区数 = max(1000/100, 1000/50) = max(10, 20) = 20

注意事项：
- 分区数只能增加不能减少（会导致 Key 路由变化）
- 过多分区的代价：
  - 每个分区占用 Broker 内存（索引缓存等）
  - Rebalance 时间增加
  - 端到端延迟增加（更多分区 = 更多 Leader 选举）
  - 文件句柄增加（每个分区 3 个文件 × 副本数）
- 经验值：单 Broker 建议 < 4000 个分区
- Kafka 3.0+ KRaft 模式支持更多分区
```

### 9.3 消息积压处理方案

```
消息积压诊断：
$ kafka-consumer-groups.sh --describe --group my-group
TOPIC    PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
orders   0          1000000         5000000         4000000  ← 积压 400 万
orders   1          1200000         5100000         3900000
orders   2          900000          4800000         3900000

处理方案（按优先级）：

1. 优化消费者处理逻辑
   - 减少单条消息处理时间
   - 异步化耗时操作（数据库写入、外部调用）
   - 批量处理替代逐条处理

2. 增加消费者实例（不超过分区数）
   当前：3 个消费者消费 6 个分区
   扩容：6 个消费者消费 6 个分区（最大并行度）

3. 增加分区数（需要评估影响）
   $ kafka-topics.sh --alter --topic orders --partitions 12
   注意：有 Key 的消息路由会变化！

4. 临时消费者 + 转发方案（紧急情况）
   原 Topic(6分区) → 临时消费者 → 新 Topic(30分区)
   → 30 个消费者并行处理
   处理完后切回原 Topic

5. 跳过非关键消息
   consumer.seek(partition, latestOffset)
   仅在消息可丢弃时使用
```

---

## 十、Go 实战代码

### 10.1 基础 Producer（使用 sarama —— 纯 Go 实现）

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/IBM/sarama"
)

// OrderEvent 订单事件
type OrderEvent struct {
	OrderID   string    `json:"order_id"`
	UserID    string    `json:"user_id"`
	Amount    float64   `json:"amount"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
}

func newSyncProducer(brokers []string) (sarama.SyncProducer, error) {
	config := sarama.NewConfig()

	// ===== 可靠性配置 =====
	config.Producer.RequiredAcks = sarama.WaitForAll // acks=all
	config.Producer.Retry.Max = 3                     // 重试 3 次
	config.Producer.Retry.Backoff = 100 * time.Millisecond
	config.Producer.Idempotent = true // 开启幂等性
	config.Net.MaxOpenRequests = 1    // 幂等模式下需要 <=5

	// ===== 性能配置 =====
	config.Producer.Compression = sarama.CompressionLZ4 // LZ4 压缩
	config.Producer.Flush.Frequency = 5 * time.Millisecond // linger.ms=5
	config.Producer.Flush.Bytes = 32 * 1024                // batch.size=32KB
	config.Producer.Flush.Messages = 100                   // 或满 100 条发送

	// ===== 必须开启 =====
	config.Producer.Return.Successes = true // SyncProducer 必须
	config.Producer.Return.Errors = true

	// ===== 分区策略 =====
	config.Producer.Partitioner = sarama.NewHashPartitioner // 按 Key 哈希

	return sarama.NewSyncProducer(brokers, config)
}

func main() {
	brokers := []string{"localhost:9092", "localhost:9093", "localhost:9094"}

	producer, err := newSyncProducer(brokers)
	if err != nil {
		log.Fatalf("Failed to create producer: %v", err)
	}
	defer producer.Close()

	// 发送订单事件
	event := OrderEvent{
		OrderID:   "ORD-20260219-001",
		UserID:    "USR-12345",
		Amount:    299.99,
		Status:    "created",
		CreatedAt: time.Now(),
	}

	value, _ := json.Marshal(event)

	msg := &sarama.ProducerMessage{
		Topic: "orders",
		Key:   sarama.StringEncoder(event.OrderID), // 同一订单发往同一分区
		Value: sarama.ByteEncoder(value),
		Headers: []sarama.RecordHeader{
			{Key: []byte("event_type"), Value: []byte("order_created")},
			{Key: []byte("source"), Value: []byte("order-service")},
		},
	}

	partition, offset, err := producer.SendMessage(msg)
	if err != nil {
		log.Fatalf("Failed to send message: %v", err)
	}
	fmt.Printf("Message sent to partition %d at offset %d\n", partition, offset)
}
```

### 10.2 高性能异步 Producer

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/IBM/sarama"
)

// AsyncProducerWrapper 封装异步 Producer，提供指标统计
type AsyncProducerWrapper struct {
	producer  sarama.AsyncProducer
	wg        sync.WaitGroup
	sent      atomic.Int64
	succeeded atomic.Int64
	failed    atomic.Int64
}

func NewAsyncProducerWrapper(brokers []string) (*AsyncProducerWrapper, error) {
	config := sarama.NewConfig()
	config.Producer.RequiredAcks = sarama.WaitForAll
	config.Producer.Compression = sarama.CompressionLZ4
	config.Producer.Flush.Frequency = 10 * time.Millisecond
	config.Producer.Flush.Bytes = 64 * 1024 // 64KB batch
	config.Producer.Flush.Messages = 500
	config.Producer.Return.Successes = true
	config.Producer.Return.Errors = true
	config.Producer.Idempotent = true
	config.Net.MaxOpenRequests = 1

	// 通道缓冲区大小
	config.ChannelBufferSize = 10000

	producer, err := sarama.NewAsyncProducer(brokers, config)
	if err != nil {
		return nil, err
	}

	w := &AsyncProducerWrapper{producer: producer}
	w.wg.Add(2)
	go w.handleSuccesses()
	go w.handleErrors()
	return w, nil
}

func (w *AsyncProducerWrapper) handleSuccesses() {
	defer w.wg.Done()
	for msg := range w.producer.Successes() {
		w.succeeded.Add(1)
		_ = msg // 可以在这里做回调处理
	}
}

func (w *AsyncProducerWrapper) handleErrors() {
	defer w.wg.Done()
	for err := range w.producer.Errors() {
		w.failed.Add(1)
		log.Printf("Producer error: topic=%s, err=%v",
			err.Msg.Topic, err.Err)
		// 生产环境：写入死信队列或告警
	}
}

func (w *AsyncProducerWrapper) Send(topic, key string, value []byte) {
	w.producer.Input() <- &sarama.ProducerMessage{
		Topic: topic,
		Key:   sarama.StringEncoder(key),
		Value: sarama.ByteEncoder(value),
	}
	w.sent.Add(1)
}

func (w *AsyncProducerWrapper) Close() {
	w.producer.AsyncClose()
	w.wg.Wait()
	fmt.Printf("Producer stats: sent=%d, succeeded=%d, failed=%d\n",
		w.sent.Load(), w.succeeded.Load(), w.failed.Load())
}

func main() {
	brokers := []string{"localhost:9092", "localhost:9093", "localhost:9094"}
	pw, err := NewAsyncProducerWrapper(brokers)
	if err != nil {
		log.Fatal(err)
	}
	defer pw.Close()

	// 高吞吐发送：100 万条消息
	start := time.Now()
	for i := 0; i < 1_000_000; i++ {
		event := map[string]interface{}{
			"id":        fmt.Sprintf("evt-%d", i),
			"timestamp": time.Now().UnixMilli(),
			"data":      "payload",
		}
		value, _ := json.Marshal(event)
		pw.Send("events", fmt.Sprintf("key-%d", i%100), value)
	}
	elapsed := time.Since(start)
	fmt.Printf("Sent 1M messages in %v (%.0f msg/s)\n",
		elapsed, 1_000_000/elapsed.Seconds())

	time.Sleep(5 * time.Second) // 等待所有消息确认
}
```

### 10.3 Consumer Group（手动提交 + 优雅关闭）

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/IBM/sarama"
)

// ConsumerGroupHandler 实现 sarama.ConsumerGroupHandler 接口
type ConsumerGroupHandler struct {
	ready    chan bool
	mu       sync.Mutex
	msgCount int64
}

// Setup 在 Rebalance 后、消费开始前调用
func (h *ConsumerGroupHandler) Setup(session sarama.ConsumerGroupSession) error {
	log.Printf("Consumer setup: memberID=%s, claims=%v",
		session.MemberID(), session.Claims())
	close(h.ready) // 通知主 goroutine 消费者已就绪
	return nil
}

// Cleanup 在 Rebalance 前调用
func (h *ConsumerGroupHandler) Cleanup(session sarama.ConsumerGroupSession) error {
	log.Printf("Consumer cleanup: memberID=%s, processed=%d messages",
		session.MemberID(), h.msgCount)
	return nil
}

// ConsumeClaim 消费分区消息的核心逻辑
func (h *ConsumerGroupHandler) ConsumeClaim(
	session sarama.ConsumerGroupSession,
	claim sarama.ConsumerGroupClaim,
) error {
	// 注意：不要在这里启动 goroutine，sarama 已经为每个 claim 启动了 goroutine
	for msg := range claim.Messages() {
		// 1. 处理消息
		if err := h.processMessage(msg); err != nil {
			log.Printf("Error processing message: topic=%s, partition=%d, offset=%d, err=%v",
				msg.Topic, msg.Partition, msg.Offset, err)
			// 处理失败策略：
			// a) 重试（有限次数）
			// b) 发送到死信队列（DLQ）
			// c) 记录并跳过
			continue
		}

		// 2. 标记消息已处理（手动提交）
		session.MarkMessage(msg, "")

		h.mu.Lock()
		h.msgCount++
		count := h.msgCount
		h.mu.Unlock()

		// 3. 每 1000 条提交一次 offset（批量提交提升性能）
		if count%1000 == 0 {
			session.Commit()
			log.Printf("Committed offset: topic=%s, partition=%d, offset=%d, total=%d",
				msg.Topic, msg.Partition, msg.Offset, count)
		}
	}
	return nil
}

func (h *ConsumerGroupHandler) processMessage(msg *sarama.ConsumerMessage) error {
	var event map[string]interface{}
	if err := json.Unmarshal(msg.Value, &event); err != nil {
		return fmt.Errorf("unmarshal error: %w", err)
	}

	// 业务处理逻辑
	// 例如：写入数据库、调用下游服务、更新缓存等
	_ = event
	return nil
}

func main() {
	brokers := []string{"localhost:9092", "localhost:9093", "localhost:9094"}
	topics := []string{"orders", "user-events"}
	groupID := "order-processing-group"

	config := sarama.NewConfig()
	config.Version = sarama.V3_6_0_0 // Kafka 3.6

	// ===== 消费者配置 =====
	config.Consumer.Group.Rebalance.GroupStrategies = []sarama.BalanceStrategy{
		sarama.NewBalanceStrategySticky(), // Sticky 分配策略
	}
	config.Consumer.Offsets.Initial = sarama.OffsetOldest // 首次从最早开始
	config.Consumer.Offsets.AutoCommit.Enable = false     // 关闭自动提交！
	config.Consumer.MaxProcessingTime = 30 * time.Second  // 单条处理超时

	// Fetch 配置
	config.Consumer.Fetch.Min = 1024           // 最小拉取 1KB
	config.Consumer.Fetch.Max = 10 * 1024 * 1024 // 最大拉取 10MB
	config.Consumer.Fetch.Default = 1024 * 1024   // 默认拉取 1MB
	config.Consumer.MaxWaitTime = 500 * time.Millisecond // 最大等待

	// Session 和心跳
	config.Consumer.Group.Session.Timeout = 30 * time.Second
	config.Consumer.Group.Heartbeat.Interval = 10 * time.Second

	client, err := sarama.NewConsumerGroup(brokers, groupID, config)
	if err != nil {
		log.Fatalf("Failed to create consumer group: %v", err)
	}
	defer client.Close()

	handler := &ConsumerGroupHandler{ready: make(chan bool)}

	ctx, cancel := context.WithCancel(context.Background())

	// 优雅关闭
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			// Consume 会在 Rebalance 时返回，需要循环调用
			if err := client.Consume(ctx, topics, handler); err != nil {
				log.Printf("Consumer error: %v", err)
				time.Sleep(time.Second) // 避免紧密循环
			}
			if ctx.Err() != nil {
				return
			}
			handler.ready = make(chan bool) // 重置 ready channel
		}
	}()

	<-handler.ready // 等待消费者就绪
	log.Println("Consumer is ready")

	<-sigChan
	log.Println("Shutting down consumer...")
	cancel()
	wg.Wait()
	log.Println("Consumer stopped")
}
```

### 10.4 Exactly Once：Consume-Transform-Produce

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/IBM/sarama"
)

// ExactlyOnceProcessor 实现 Consume-Transform-Produce 的 Exactly Once 语义
type ExactlyOnceProcessor struct {
	consumerGroup sarama.ConsumerGroup
	producer      sarama.AsyncProducer
	sourceTopic   string
	sinkTopic     string
}

func NewExactlyOnceProcessor(brokers []string, groupID, source, sink string) (*ExactlyOnceProcessor, error) {
	// Producer 配置（事务模式）
	pConfig := sarama.NewConfig()
	pConfig.Version = sarama.V3_6_0_0
	pConfig.Producer.RequiredAcks = sarama.WaitForAll
	pConfig.Producer.Idempotent = true
	pConfig.Producer.Return.Successes = true
	pConfig.Producer.Return.Errors = true
	pConfig.Net.MaxOpenRequests = 1
	// 注意：sarama 对事务支持有限，生产环境建议用 confluent-kafka-go
	// 这里展示概念性实现

	producer, err := sarama.NewAsyncProducer(brokers, pConfig)
	if err != nil {
		return nil, fmt.Errorf("create producer: %w", err)
	}

	// Consumer 配置
	cConfig := sarama.NewConfig()
	cConfig.Version = sarama.V3_6_0_0
	cConfig.Consumer.Group.Rebalance.GroupStrategies = []sarama.BalanceStrategy{
		sarama.NewBalanceStrategySticky(),
	}
	cConfig.Consumer.Offsets.AutoCommit.Enable = false
	cConfig.Consumer.Offsets.Initial = sarama.OffsetOldest

	consumer, err := sarama.NewConsumerGroup(brokers, groupID, cConfig)
	if err != nil {
		producer.Close()
		return nil, fmt.Errorf("create consumer: %w", err)
	}

	return &ExactlyOnceProcessor{
		consumerGroup: consumer,
		producer:      producer,
		sourceTopic:   source,
		sinkTopic:     sink,
	}, nil
}

// TransformHandler 实现消费-转换-生产
type TransformHandler struct {
	producer  sarama.AsyncProducer
	sinkTopic string
}

func (h *TransformHandler) Setup(_ sarama.ConsumerGroupSession) error   { return nil }
func (h *TransformHandler) Cleanup(_ sarama.ConsumerGroupSession) error { return nil }

func (h *TransformHandler) ConsumeClaim(
	session sarama.ConsumerGroupSession,
	claim sarama.ConsumerGroupClaim,
) error {
	for msg := range claim.Messages() {
		// 1. 反序列化源消息
		var input map[string]interface{}
		if err := json.Unmarshal(msg.Value, &input); err != nil {
			log.Printf("Skip invalid message: offset=%d, err=%v", msg.Offset, err)
			session.MarkMessage(msg, "")
			continue
		}

		// 2. 转换逻辑（业务处理）
		output := h.transform(input)

		// 3. 序列化并发送到目标 Topic
		value, _ := json.Marshal(output)
		h.producer.Input() <- &sarama.ProducerMessage{
			Topic: h.sinkTopic,
			Key:   sarama.ByteEncoder(msg.Key),
			Value: sarama.ByteEncoder(value),
			Headers: []sarama.RecordHeader{
				{Key: []byte("source_topic"), Value: []byte(msg.Topic)},
				{Key: []byte("source_partition"), Value: []byte(fmt.Sprintf("%d", msg.Partition))},
				{Key: []byte("source_offset"), Value: []byte(fmt.Sprintf("%d", msg.Offset))},
			},
		}

		// 4. 提交 offset
		// 注意：真正的 Exactly Once 需要事务将 produce 和 offset commit 原子化
		// sarama 的事务支持有限，生产环境用 confluent-kafka-go 或 Java 客户端
		session.MarkMessage(msg, "")
		session.Commit()
	}
	return nil
}

func (h *TransformHandler) transform(input map[string]interface{}) map[string]interface{} {
	// 示例：数据清洗和enrichment
	output := make(map[string]interface{})
	output["original"] = input
	output["processed_at"] = time.Now().UnixMilli()
	output["version"] = "v2"

	// 数据清洗：去除空值
	for k, v := range input {
		if v != nil && v != "" {
			output[k] = v
		}
	}
	return output
}

func main() {
	brokers := []string{"localhost:9092", "localhost:9093", "localhost:9094"}

	processor, err := NewExactlyOnceProcessor(
		brokers,
		"transform-group",
		"raw-events",      // 源 Topic
		"processed-events", // 目标 Topic
	)
	if err != nil {
		log.Fatal(err)
	}

	handler := &TransformHandler{
		producer:  processor.producer,
		sinkTopic: processor.sinkTopic,
	}

	ctx := context.Background()
	for {
		if err := processor.consumerGroup.Consume(ctx, []string{processor.sourceTopic}, handler); err != nil {
			log.Printf("Consumer error: %v", err)
			time.Sleep(time.Second)
		}
	}
}
```

### 10.5 生产级 Kafka 管理工具

```go
package main

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"time"

	"github.com/IBM/sarama"
)

// KafkaAdmin Kafka 集群管理工具
type KafkaAdmin struct {
	admin  sarama.ClusterAdmin
	client sarama.Client
}

func NewKafkaAdmin(brokers []string) (*KafkaAdmin, error) {
	config := sarama.NewConfig()
	config.Version = sarama.V3_6_0_0

	admin, err := sarama.NewClusterAdmin(brokers, config)
	if err != nil {
		return nil, err
	}

	client, err := sarama.NewClient(brokers, config)
	if err != nil {
		admin.Close()
		return nil, err
	}

	return &KafkaAdmin{admin: admin, client: client}, nil
}

// CreateTopic 创建 Topic（生产级配置）
func (ka *KafkaAdmin) CreateTopic(name string, partitions int32, replication int16) error {
	detail := sarama.TopicDetail{
		NumPartitions:     partitions,
		ReplicationFactor: replication,
		ConfigEntries: map[string]*string{
			"cleanup.policy":      strPtr("delete"),
			"retention.ms":        strPtr("604800000"),  // 7 天
			"segment.bytes":       strPtr("1073741824"), // 1GB
			"min.insync.replicas": strPtr("2"),
			"compression.type":    strPtr("lz4"),
		},
	}
	return ka.admin.CreateTopic(name, &detail, false)
}

// GetConsumerGroupLag 获取消费者组的消费延迟
func (ka *KafkaAdmin) GetConsumerGroupLag(groupID string) (map[string]map[int32]int64, error) {
	// 获取消费者组的 offset
	offsets, err := ka.admin.ListConsumerGroupOffsets(groupID, nil)
	if err != nil {
		return nil, err
	}

	lag := make(map[string]map[int32]int64)

	for topic, partitions := range offsets.Blocks {
		lag[topic] = make(map[int32]int64)
		for partition, block := range partitions {
			// 获取分区的最新 offset
			latestOffset, err := ka.client.GetOffset(topic, partition, sarama.OffsetNewest)
			if err != nil {
				continue
			}
			lag[topic][partition] = latestOffset - block.Offset
		}
	}
	return lag, nil
}

// PrintClusterInfo 打印集群信息
func (ka *KafkaAdmin) PrintClusterInfo() {
	brokers := ka.client.Brokers()
	fmt.Printf("=== Kafka Cluster Info ===\n")
	fmt.Printf("Brokers: %d\n", len(brokers))
	for _, b := range brokers {
		fmt.Printf("  - ID=%d, Addr=%s\n", b.ID(), b.Addr())
	}

	topics, _ := ka.client.Topics()
	sort.Strings(topics)
	fmt.Printf("\nTopics: %d\n", len(topics))
	for _, t := range topics {
		if strings.HasPrefix(t, "__") {
			continue // 跳过内部 Topic
		}
		partitions, _ := ka.client.Partitions(t)
		fmt.Printf("  - %s (%d partitions)\n", t, len(partitions))
	}
}

// MonitorLag 持续监控消费延迟
func (ka *KafkaAdmin) MonitorLag(ctx context.Context, groupID string, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			lag, err := ka.GetConsumerGroupLag(groupID)
			if err != nil {
				log.Printf("Error getting lag: %v", err)
				continue
			}

			totalLag := int64(0)
			for topic, partitions := range lag {
				for partition, l := range partitions {
					totalLag += l
					if l > 10000 { // 告警阈值
						log.Printf("⚠️  HIGH LAG: topic=%s, partition=%d, lag=%d",
							topic, partition, l)
					}
				}
			}
			log.Printf("Total lag for group %s: %d", groupID, totalLag)
		}
	}
}

func strPtr(s string) *string { return &s }

func main() {
	brokers := []string{"localhost:9092", "localhost:9093", "localhost:9094"}
	admin, err := NewKafkaAdmin(brokers)
	if err != nil {
		log.Fatal(err)
	}

	// 打印集群信息
	admin.PrintClusterInfo()

	// 创建 Topic
	if err := admin.CreateTopic("orders", 6, 3); err != nil {
		log.Printf("Create topic: %v", err)
	}

	// 监控消费延迟
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	admin.MonitorLag(ctx, "order-processing-group", 10*time.Second)
}
```

---

## 十一、Docker Compose 快速搭建 Kafka 集群

### 11.1 KRaft 模式 3 节点集群

```yaml
# docker-compose-kafka.yml
version: '3.8'

services:
  kafka-1:
    image: apache/kafka:3.7.0
    container_name: kafka-1
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_MIN_INSYNC_REPLICAS: 2
      KAFKA_DEFAULT_REPLICATION_FACTOR: 3
      KAFKA_NUM_PARTITIONS: 6
      CLUSTER_ID: 'MkU3OEVBNTcwNTJENDM2Qk'
    volumes:
      - kafka-1-data:/var/lib/kafka/data

  kafka-2:
    image: apache/kafka:3.7.0
    container_name: kafka-2
    ports:
      - "9094:9094"
    environment:
      KAFKA_NODE_ID: 2
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9094,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9094
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_MIN_INSYNC_REPLICAS: 2
      CLUSTER_ID: 'MkU3OEVBNTcwNTJENDM2Qk'
    volumes:
      - kafka-2-data:/var/lib/kafka/data

  kafka-3:
    image: apache/kafka:3.7.0
    container_name: kafka-3
    ports:
      - "9096:9096"
    environment:
      KAFKA_NODE_ID: 3
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9096,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9096
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_MIN_INSYNC_REPLICAS: 2
      CLUSTER_ID: 'MkU3OEVBNTcwNTJENDM2Qk'
    volumes:
      - kafka-3-data:/var/lib/kafka/data

  # Kafka UI 管理界面
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: kafka-ui
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka-1:9092,kafka-2:9094,kafka-3:9096
    depends_on:
      - kafka-1
      - kafka-2
      - kafka-3

volumes:
  kafka-1-data:
  kafka-2-data:
  kafka-3-data:
```

```bash
# 启动集群
docker compose -f docker-compose-kafka.yml up -d

# 创建 Topic
docker exec kafka-1 kafka-topics.sh --create \
  --topic orders --partitions 6 --replication-factor 3 \
  --bootstrap-server localhost:9092

# 查看 Topic 详情
docker exec kafka-1 kafka-topics.sh --describe \
  --topic orders --bootstrap-server localhost:9092

# 生产消息测试
docker exec -it kafka-1 kafka-console-producer.sh \
  --topic orders --bootstrap-server localhost:9092

# 消费消息测试
docker exec -it kafka-1 kafka-console-consumer.sh \
  --topic orders --from-beginning --group test-group \
  --bootstrap-server localhost:9092

# 查看消费者组
docker exec kafka-1 kafka-consumer-groups.sh \
  --describe --group test-group --bootstrap-server localhost:9092

# 性能测试：Producer
docker exec kafka-1 kafka-producer-perf-test.sh \
  --topic perf-test --num-records 1000000 --record-size 1024 \
  --throughput -1 --producer-props bootstrap.servers=localhost:9092 \
  acks=all compression.type=lz4

# 性能测试：Consumer
docker exec kafka-1 kafka-consumer-perf-test.sh \
  --topic perf-test --messages 1000000 \
  --bootstrap-server localhost:9092
```

---

## 十二、Kafka 在数据 Pipeline 中的典型架构

### 12.1 CDC 数据同步架构

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│  MySQL   │───→│ Debezium │───→│  Kafka   │───→│Elasticsearch │
│ (Source) │CDC │ Connect  │    │  Topic   │    │  (Search)    │
└──────────┘    └──────────┘    │          │    └──────────────┘
                                │          │───→┌──────────────┐
                                │          │    │    TiDB      │
                                │          │    │ (Analytics)  │
                                └──────────┘    └──────────────┘

数据流：
1. MySQL 产生 Binlog（Row 格式）
2. Debezium Source Connector 捕获 Binlog 变更
3. 变更事件写入 Kafka Topic（每张表一个 Topic）
4. 下游 Sink Connector 消费并写入目标系统
5. 支持全量快照 + 增量同步
```

### 12.2 实时数据处理 Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ App Logs │───→│  Kafka   │───→│  Flink   │───→│  Kafka   │
│ Metrics  │    │ (Raw)    │    │ (Clean)  │    │(Cleaned) │
│ Events   │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                     │
                                    ┌────────────────┼────────────┐
                                    ↓                ↓            ↓
                              ┌──────────┐    ┌──────────┐  ┌─────────┐
                              │   ES     │    │  ClickHouse│  │  S3    │
                              │ (Search) │    │ (OLAP)   │  │(Archive)│
                              └──────────┘    └──────────┘  └─────────┘
```

---

## 十三、面试高频问题与深度回答

### Q1：Kafka 如何保证消息不丢失？

```
三端保证：

Producer 端：
  acks=all + retries + enable.idempotence=true
  → 消息写入所有 ISR 副本才算成功
  → 网络失败自动重试
  → 幂等性防止重试导致重复

Broker 端：
  replication.factor >= 3
  min.insync.replicas >= 2
  unclean.leader.election.enable=false
  → 至少 3 副本
  → 至少 2 个副本同步才允许写入
  → ISR 全挂宁可不可用也不丢数据

Consumer 端：
  enable.auto.commit=false
  → 手动提交 offset
  → 处理成功后再提交
  → 崩溃后从上次提交的 offset 重新消费（At Least Once）
```

### Q2：Kafka 和 RabbitMQ/Pulsar 的区别？

```
┌──────────┬──────────────┬──────────────┬──────────────┐
│ 特性      │ Kafka        │ RabbitMQ     │ Pulsar       │
├──────────┼──────────────┼──────────────┼──────────────┤
│ 模型      │ 发布-订阅    │ 队列+交换机   │ 发布-订阅    │
│ 存储      │ 日志追加     │ 内存+磁盘    │ BookKeeper   │
│ 吞吐      │ 百万/s       │ 万级/s       │ 百万/s       │
│ 延迟      │ ms级         │ μs级         │ ms级         │
│ 消息回溯  │ ✅ 支持       │ ❌ 不支持     │ ✅ 支持      │
│ 顺序保证  │ 分区内有序   │ 队列内有序    │ 分区内有序   │
│ 计算存储  │ 耦合         │ 耦合         │ 分离         │
│ 多租户    │ ❌           │ ❌           │ ✅ 原生支持   │
│ 地域复制  │ MirrorMaker  │ Federation   │ 原生支持     │
│ 运维复杂度│ 中等         │ 低           │ 高           │
│ 适用场景  │ 大数据/流处理│ 业务消息队列  │ 云原生/多租户│
└──────────┴──────────────┴──────────────┴──────────────┘
```

### Q3：如何设计一个高可用的 Kafka 集群？

```
生产环境最佳实践：

1. 集群规模：
   - 最少 3 个 Broker（满足 3 副本）
   - KRaft Controller 3-5 个节点
   - 建议 Broker 和 Controller 分离部署

2. Topic 配置：
   - replication.factor=3
   - min.insync.replicas=2
   - acks=all
   - 分区数 = 预期吞吐 / 单分区吞吐

3. 跨机架/可用区部署：
   broker.rack=rack-1  # 每个 Broker 标记机架
   → Kafka 自动将副本分散到不同机架
   → 单机架故障不影响可用性

4. 监控告警：
   - ISR 收缩告警（Under-Replicated Partitions > 0）
   - 消费延迟告警（Consumer Lag > 阈值）
   - Broker 磁盘使用率告警（> 80%）
   - Controller 切换告警
   - 网络吞吐告警

5. 容量规划：
   磁盘 = 日消息量 × 消息大小 × 副本数 × 保留天数 × 1.2（余量）
   示例：1亿条/天 × 1KB × 3副本 × 7天 × 1.2 = ~2.5TB
```

### Q4：Kafka 消息积压了 1000 万条，怎么处理？

```
紧急处理流程：

Step 1: 诊断（5分钟）
  $ kafka-consumer-groups.sh --describe --group xxx
  → 确认哪些分区积压、积压量、消费速度

Step 2: 判断原因
  a) 消费者处理慢 → 优化处理逻辑
  b) 消费者挂了 → 重启消费者
  c) 下游服务故障 → 修复下游
  d) 流量突增 → 扩容

Step 3: 快速恢复
  方案A（推荐）：扩容消费者到分区数
  方案B：临时增加分区 + 消费者
  方案C：消息转发到更多分区的临时 Topic
  方案D（最后手段）：跳过非关键消息

Step 4: 事后复盘
  - 设置消费延迟告警（提前发现）
  - 优化消费者处理性能
  - 评估是否需要增加分区数
  - 考虑消费者自动扩缩容
```

---

## 学习笔记

### 核心理解
1. Kafka 的高性能来自四大设计：顺序写、零拷贝、页缓存、批处理+压缩
2. 消息可靠性是 Producer（acks=all）+ Broker（多副本+ISR）+ Consumer（手动提交）三端协同
3. Exactly Once 语义通过幂等 Producer + 事务实现，但有 ~20% 性能开销
4. KRaft 模式是 Kafka 的未来，去掉 ZooKeeper 依赖，支持百万级分区
5. 分区数是并行度的上限，消费者数量不应超过分区数

### 与 JD 的对应关系
- **数据采集 Pipeline**：Kafka Connect + Debezium 实现 CDC
- **数据清洗去重**：Kafka Streams 或 Flink 消费 Kafka 做流式处理
- **性能与吞吐优化**：Producer 批处理、压缩、分区策略调优
- **高可用架构**：多副本、ISR、KRaft、跨机架部署
- **分布式系统可靠性**：Exactly Once 语义、事务消息、消费者组管理

### 后续行动计划
1. 用 Docker Compose 搭建 KRaft 模式 3 节点集群
2. 用 Go (sarama) 实现完整的 Producer/Consumer
3. 搭建 Debezium + Kafka Connect 的 CDC Pipeline
4. 做 Producer/Consumer 性能压测，对比不同配置的吞吐差异
5. 学习 Kafka Streams 实现简单的流处理应用
