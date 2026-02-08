# 分布式存储与消息系统复习（4天）

## 概述
- **目标**：系统复习分布式存储与消息系统的核心原理与实践，重点掌握 MySQL/TiDB/Redis/Kafka/Pulsar 等关键技术的架构设计、性能调优与高可用方案，满足JD中"高并发服务端与API系统"和"大规模数据处理Pipeline"领域的数据存储与消息传输要求
- **时间**：春节第2周中间4天（可与其他路径并行学习）
- **前提**：熟悉SQL与基本数据结构，有数据库使用经验，需要系统化深入理解
- **强度**：高强度（每天8-10小时），适合需要快速提升数据系统能力的工程师

## JD要求对应

### JD领域覆盖
| JD领域 | 对应内容 | 优先级 |
|--------|----------|--------|
| 一、高并发服务端与API系统 | MySQL性能优化、Redis缓存策略、数据库连接池、分布式事务 | ⭐⭐⭐ |
| 二、大规模数据处理Pipeline | Kafka/Pulsar消息队列、TiDB分布式存储、数据分片与复制 | ⭐⭐⭐ |
| 三、Agent基础设施与运行时平台 | 数据存储选型、消息系统集成、状态持久化 | ⭐⭐ |
| 四、异构超算基础设施 | 分布式协调、数据一致性、高可用存储 | ⭐⭐ |

### JD能力对应
| 能力要求 | 学习内容 | 验证方式 |
|----------|----------|----------|
| **数据库原理与调优** | MySQL索引机制、查询优化、锁机制、事务隔离级别 | 数据库优化方案 |
| **分布式系统深刻理解** | TiDB分布式架构、数据分片、Raft一致性协议 | 分布式存储方案 |
| **高可用高可靠架构** | MySQL主从复制、Redis Cluster/Sentinel、Kafka集群高可用 | 高可用架构设计 |
| **大数据处理经验** | Kafka分区策略、Pulsar分层存储、消息积压处理 | 数据流处理设计 |
| **性能分析与Profiling** | 慢查询分析、性能监控、压测与调优 | 性能优化报告 |

## 学习重点

### 1. MySQL深度解析（第1天上午）
**JD引用**："对数据库原理有深入理解，拥有丰富的性能调优与大数据处理经验"

**核心内容**：
- MySQL架构与存储引擎（InnoDB vs MyISAM、Memory、CSV）
- 索引原理与优化（B+树索引、哈希索引、全文索引、函数索引）
- 事务ACID与隔离级别（READ UNCOMMITTED到SERIALIZABLE）
- MVCC多版本并发控制原理
- 锁机制（行锁、间隙锁、临键锁、意向锁）
- 查询优化器与执行计划分析
- 分区表与分库分表策略
- 主从复制原理（基于语句、基于行、混合）
- MySQL Group Replication与InnoDB Cluster
- 备份与恢复（物理备份、逻辑备份、增量备份）

**实践任务**：
- 分析慢查询日志，优化复杂SQL查询
- 设计分库分表方案，评估分片键选择
- 配置MySQL主从复制与故障切换
- 实现读写分离中间件方案
- 设计数据迁移与回滚方案

### 2. MySQL性能调优（第1天下午）
**JD引用**："负责核心服务的性能优化、数据库调优与分布式系统可靠性保障"

**核心内容**：
- 查询性能优化（EXPLAIN分析、索引覆盖、索引下推）
- 表结构设计优化（数据类型选择、范式与反范式）
- 连接池配置与优化（连接数、超时、空闲连接）
- 缓冲池配置与优化（Buffer Pool、Change Buffer、Adaptive Hash）
- 性能监控指标（QPS、TPS、连接数、缓冲池命中率）
- Performance Schema与Sys Schema使用
- 慢查询分析与优化策略
- 批量操作与事务优化
- 死锁检测与避免
- MySQL 8.0新特性（窗口函数、CTE、降序索引、隐藏列）

**实践任务**：
- 使用EXPLAIN分析并优化10个复杂查询
- 配置MySQL性能监控（Prometheus + Grafana）
- 设计连接池优化方案（HikariCP配置）
- 实现批量数据导入优化策略
- 编写性能测试报告与优化建议

### 3. Redis深度应用（第2天上午）
**JD引用**："构建高性能、高安全性的Agent运行时环境"

**核心内容**：
- Redis数据结构与底层实现（SDS、跳表、压缩列表、整数集合）
- 持久化机制（RDB快照、AOF日志、混合持久化）
- 主从复制与故障转移
- 哨兵模式（Sentinel）高可用架构
- Redis Cluster集群架构与分片原理
- 缓存策略（Cache Aside、Read Through、Write Through）
- 缓存问题解决方案（穿透、击穿、雪崩、一致性）
- 分布式锁实现（Redlock、SET NX、Lua脚本）
- 管道、事务、Lua脚本与Pub/Sub
- 内存优化与淘汰策略
- Redis 7.0新特性（Functions、ACL、Multi-key操作）

**实践任务**：
- 设计多级缓存架构（本地缓存 + Redis）
- 实现分布式锁（Redlock算法）
- 配置Redis Cluster集群与故障测试
- 设计缓存预热与更新策略
- 解决缓存一致性问题（延迟双删、消息通知）

### 4. 缓存架构与性能优化（第2天下午）
**JD引用**："负责核心服务的性能优化"

**核心内容**：
- 缓存架构模式（Cache-Aside、Read-Through、Write-Through、Write-Behind）
- 热点数据识别与处理
- 缓存预热策略
- 缓存更新策略（主动更新、被动淘汰、定时刷新）
- 大Key与热Key问题解决方案
- 缓存分片策略（一致性哈希、虚拟桶）
- 本地缓存选择（Caffeine、Guava Cache、Ehcache）
- 缓存监控与告警（命中率、内存使用、QPS）
- Redis性能优化（Pipeline、批量操作、连接池）
- 缓存容量规划与成本优化

**实践任务**：
- 设计多级缓存方案（L1本地 + L2分布式）
- 实现热点数据自动发现与处理
- 配置缓存监控告警（Prometheus + Grafana）
- 设计缓存降级与限流策略
- 编写缓存最佳实践文档

### 5. Kafka深度解析（第3天上午）
**JD引用**："负责数据采集、清洗、去重与质量评估系统的设计与开发"

**核心内容**：
- Kafka架构（Broker、Topic、Partition、Segment）
- 消息存储机制（日志段、索引文件、稀疏索引）
- 生产者核心机制（批处理、压缩、重试、幂等）
- 消费者组与再平衡（Rebalance协议、Sticky分配）
- 消息投递语义（At Most Once、At Least Once、Exactly Once）
- 高可用与容错（ISR、副本同步、Leader选举）
- Kafka Connect与Kafka Streams
- 顺序保证与消息积压处理
- 事务消息实现原理
- Kafka 3.0+新特性（KRaft模式、增量再平衡）

**实践任务**：
- 搭建Kafka集群（3节点、多副本）
- 实现 Exactly Once 语义消费者
- 设计消息积压处理方案
- 配置Kafka监控（Kafka Exporter + Prometheus）
- 实现事务消息生产者

### 6. Pulsar架构与实践（第3天下午）
**JD引用**："构建服务于搜索、多模态与模型训练的高质量数据湖与索引系统"

**核心内容**：
- Pulsar架构（Broker、BookKeeper、Zookeeper/元数据存储）
- 存储分离架构（计算存储分离、分层存储）
- 订阅模型（Exclusive、Failover、Shared、Key_Shared）
- 消息去重与消息TTL
- 多地域复制（Geo-Replication）
- Functions与Pulsar IO
- 分层存储（S3、HDFS、GCS冷存储）
- Pulsar与Kafka架构对比
- Pulsar 2.11+新特性（Transaction、Protocol Handlers）
- Pulsar安全（TLS加密、ACL认证、授权）

**实践任务**：
- 搭建Pulsar集群（Standalone或Docker）
- 实现多地域复制方案
- 配置分层存储（热数据BookKeeper + 冷数据S3）
- 设计Pulsar Functions数据处理链路
- 对比Kafka与Pulsar选型决策

### 7. TiDB分布式数据库（第4天上午）
**JD引用**："对数据库原理有深入理解，拥有丰富的性能调优与大数据处理经验"

**核心内容**：
- TiDB架构（PD、TiDB、TiKV、TiFlash）
- TiKV存储引擎（RocksDB、Raft、MVCC）
- PD调度器与元数据管理
- 分布式事务（两阶段提交、乐观锁、悲观锁）
- 数据分片与路由策略
- HTAP混合事务分析处理（TiFlash列式存储）
- SQL优化与统计信息收集
- TiFlash实时分析引擎
- 数据迁移工具（DM、TiDB Data Migration）
- TiDB Cloud与Serverless架构
- TiDB 7.0+新特性（资源管控、PaTiEngine）

**实践任务**：
- 使用Docker Compose部署TiDB集群
- 分析TiDB慢查询与优化SQL
- 设计HTAP方案（TP业务 + AP分析）
- 实现数据迁移方案（MySQL到TiDB）
- 配置TiDB监控（Grafana Dashboard）

### 8. 数据系统架构设计与总结（第4天下午）
**JD引用**："能够设计高可用、高可靠的系统架构"

**核心内容**：
- 数据存储选型决策树（SQL vs NoSQL、关系型 vs 文档型 vs 图数据库）
- 多模型数据架构（Polyglot Persistence）
- 数据一致性方案（强一致性、最终一致性、因果一致性）
- 分布式事务模式（Saga、TCC、本地消息表、事务消息）
- 数据网格架构（Data Mesh）
- 数据湖仓一体（Lakehouse）
- 实时数据处理架构（Lambda vs Kappa vs Modern Data Stack）
- 数据治理与数据质量
- 数据安全与合规（加密、脱敏、审计）
- 成本优化与容量规划

**实践任务**：
- 设计多模型数据架构方案
- 设计分布式事务协调方案（Saga模式）
- 设计实时数据处理Pipeline
- 编写数据架构决策文档（ADR）
- 总结数据系统设计最佳实践

## 实践项目：高性能数据平台设计

### 项目目标
**JD对应**：满足"高并发服务端与API系统"和"大规模数据处理Pipeline"的数据存储与消息传输要求

设计一个生产级高性能数据平台的核心架构，包含：
1. 多级缓存系统（本地缓存 + Redis Cluster）
2. 读写分离的MySQL主从架构
3. TiDB分布式数据库作为数据分析层
4. Kafka消息队列用于异步处理
5. Pulsar作为事件流处理引擎
6. 完整的监控与可观测性系统

### 技术栈参考（明确版本）
- **关系型数据库**：MySQL 8.0+（InnoDB引擎）
- **分布式数据库**：TiDB 7.1+
- **缓存**：Redis 7.0+ Cluster / Caffeine（本地缓存）
- **消息队列**：Apache Kafka 3.6+ / Apache Pulsar 3.0+
- **连接池**：HikariCP 5.0+ / JDBC
- **监控**：Prometheus 2.45+ + Grafana 10.0+
- **数据迁移**：TiDB DM、Debezium 2.4+
- **客户端**：Redis Lettuce 6.2+ / Kafka 3.5+ / Pulsar Client 3.0+

### 环境配置要求
- **本地开发环境**：
  - Docker 24.0+ / Docker Compose 2.20+
  - Java 17+ / Go 1.21+ / Python 3.11+
  - Maven 3.9+ / Gradle 8.0+ / Go Modules

- **依赖安装**：
  ```bash
  # 启动MySQL主从集群
  docker-compose -f docker/mysql-cluster.yml up -d

  # 启动Redis Cluster
  docker-compose -f docker/redis-cluster.yml up -d

  # 启动Kafka集群
  docker-compose -f docker/kafka-cluster.yml up -d

  # 启动TiDB集群
  docker-compose -f docker/tidb-cluster.yml up -d

  # 启动Pulsar集群
  docker-compose -f docker/pulsar-cluster.yml up -d
  ```

### 架构设计
```
high-performance-data-platform/
├── cache-layer/           # 多级缓存层
│   ├── local-cache/       # Caffeine/Guava本地缓存
│   ├── redis-cluster/     # Redis Cluster集群
│   └── cache-coordinator/ # 缓存一致性协调器
├── rdbms-layer/           # 关系型数据库层
│   ├── mysql-master/      # MySQL主节点
│   ├── mysql-slave/       # MySQL从节点（读）
│   ├── proxy-sql/         # 读写分离代理
│   └── orchestrator/      # 主从拓扑管理
├── distributed-db/        # 分布式数据库层
│   ├── tidb-server/       # TiDB计算节点
│   ├── tikv-nodes/        # TiKV存储节点
│   ├── pd-nodes/          # PD调度节点
│   └── tiflash-nodes/     # TiFlash分析节点
├── messaging-layer/       # 消息队列层
│   ├── kafka-cluster/     # Kafka集群（持久化消息）
│   ├── pulsar-cluster/    # Pulsar集群（事件流）
│   └── connector/         # Kafka Connect / Pulsar IO
├── data-sync/             # 数据同步层
│   ├── debezium/          # CDC数据变更捕获
│   ├── data-migration/    # TiDB DM数据迁移
│   └── schema-sync/       # 表结构同步工具
└── observability/         # 可观测性层
    ├── metrics/           # Prometheus指标采集
    ├── logs/              # 日志聚合（ELK）
    ├── tracing/           # 分布式追踪
    └── alerting/          # 告警规则
```

### 核心组件设计

#### 1. 多级缓存架构
```yaml
架构模式：
  L1本地缓存: Caffeine (堆内内存)
    ├── 容量: 10K条记录
    ├── 过期: 5分钟TTL
    └── 淘汰: W-TinyLFU策略

  L2分布式缓存: Redis Cluster (堆外内存)
    ├── 节点: 6节点（3主3从）
    ├── 分片: 16384个槽
    ├── 持久化: AOF + RDB混合
    └── 淘汰: allkeys-lru

缓存策略：
  - Cache-Aside模式（读穿透）
  - Write-Through模式（双写）
  - 延迟双删保证一致性
  - 布隆过滤器防缓存穿透
```

#### 2. MySQL主从架构
```yaml
架构模式：
  MySQL Master (1节点)
    ├── 写流量: 全部写入
    ├── Binlog: Row格式
    └── GTID: 启用

  MySQL Slave (2节点)
    ├── 读流量: 只读查询
    ├── 复制模式: 半同步复制
    └── 延迟监控: <1秒

  ProxySQL中间件
    ├── 读写分离规则
    ├── 连接池管理
    └── 查询缓存
```

#### 3. TiDB HTAP架构
```yaml
业务场景：
  - TP业务: MySQL处理OLTP事务
  - AP业务: TiDB/TiFlash处理OLAP分析

数据同步：
  MySQL → (CDC) → TiDB
    ├── Debezium捕获Binlog
    ├── Kafka消息队列缓冲
    └── TiDB DM实时同步

查询路由：
  - 实时查询: TiDB（行存）
  - 分析查询: TiFlash（列存）
  - 历史查询: S3冷存储
```

#### 4. Kafka消息架构
```yaml
Topic设计：
  - 订单事件: orders（12分区，3副本）
  - 用户行为: user-events（24分区，3副本）
  - 数据变更: cdc-events（6分区，3副本）

消费者组：
  - 订单处理组: 3个消费者实例
  - 数据分析组: 5个消费者实例

配置优化：
  - batch.size: 32KB
  - linger.ms: 10ms
  - compression.type: lz4
  - acks: all
```

#### 5. Pulsar事件流架构
```yaml
租户设计：
  - 生产租户: prod（资源隔离）
  - 分析租户: analytics（资源隔离）

命名空间：
  - prod/events（事件流）
  - analytics/streams（分析流）

分层存储：
  - 热数据: BookKeeper（7天）
  - 冷数据: S3（永久）

订阅模式：
  - 故障转移: Exclusive（订单处理）
  - 负载均衡: Shared（日志处理）
```

## 学习资源

### 经典书籍
1. **《高性能MySQL》**（第4版）：MySQL性能优化圣经
2. **《MySQL技术内幕：InnoDB存储引擎》**：深入理解InnoDB原理
3. **《Redis设计与实现》**：Redis底层原理详解
4. **《Kafka权威指南》**（第2版）：Kafka全面指南
5. **《Designing Data-Intensive Applications》**（DDIA）：数据密集型应用设计
6. **《Database Internals》**：数据库内部实现原理
7. **《Streaming Systems》**：流处理系统设计

### 官方文档
1. **MySQL官方文档**：[MySQL 8.0 Reference Manual](https://dev.mysql.com/doc/refman/8.0/en/)
2. **TiDB文档**：[TiDB Documentation](https://docs.pingcap.com/tidb/stable)
3. **Redis文档**：[Redis Documentation](https://redis.io/documentation)
4. **Kafka文档**：[Apache Kafka Documentation](https://kafka.apache.org/documentation/)
5. **Pulsar文档**：[Apache Pulsar Documentation](https://pulsar.apache.org/docs/)
6. **Percona Blog**：MySQL性能优化最佳实践

### 在线课程
1. **CMU 15-445**：[数据库系统](https://www.youtube.com/playlist?list=PLSE8ODhjZXjYtlRh4pA0FbhTRRndQLdWq)
2. **Stanford CS346**：[数据库系统](https://web.stanford.edu/class/cs346/)
3. **MIT 6.824**：[分布式系统](https://pdos.csail.mit.edu/6.824/) - Raft论文与实现
4. **Kafka官方教程**：[Confluent Kafka Training](https://developer.confluent.io/)
5. **Redis University**：[Redis官方课程](https://university.redis.com/)

### 技术博客与案例
1. **Uber Engineering**：[Schemaless与数据分片](https://eng.uber.com/schemaless-part-one/)
2. **Netflix Tech Blog**：[EVCache缓存架构](https://netflixtechblog.com/evcache-optimized-caching-for-the-cloud-ca80d6d6e741)
3. **Instagram Engineering**：[Sharding与数据存储](https://instagram-engineering.com/what-powers-instagram-hundreds-of-instances-dozens-of-technologies-adf2e22da2ad)
4. **LinkedIn Engineering**：[Kafka与数据流](https://engineering.linkedin.com/kafka/)
5. **TiDB Blog**：[分布式数据库实践](https://www.pingcap.com/blog/)
6. **Cloudflare Blog**：[Redis缓存架构](https://blog.cloudflare.com/)

### 开源项目参考
1. **MySQL**：[github.com/mysql/mysql-server](https://github.com/mysql/mysql-server) - MySQL源码
2. **TiDB**：[github.com/pingcap/tidb](https://github.com/pingcap/tidb) - 分布式数据库
3. **Redis**：[github.com/redis/redis](https://github.com/redis/redis) - Redis源码
4. **Apache Kafka**：[kafka.apache.org](https://kafka.apache.org/) - 分布式消息队列
5. **Apache Pulsar**：[pulsar.apache.org](https://pulsar.apache.org/) - 云原生消息平台
6. **Debezium**：[github.com/debezium/debezium](https://github.com/debezium/debezium) - CDC数据变更捕获
7. **HikariCP**：[github.com/brettwooldridge/HikariCP](https://github.com/brettwooldridge/HikariCP) - JDBC连接池
8. **Caffeine**：[github.com/ben-manes/caffeine](https://github.com/ben-manes/caffeine) - 高性能缓存

### 权威论文
1. **The Log: What every software engineer should know about real-time data's unifying abstraction** (Jay Kreps, 2013)
2. **Kafka: a Distributed Messaging System for Log Processing** (Kafka论文, 2011)
3. **Dynamo: Amazon's Highly Available Key-value Store** (2007)
4. **Google Spanner: Google's Globally-Distributed Database** (2012)
5. **Raft: In Search of an Understandable Consensus Algorithm** (2014)
6. **Designing Data-Intensive Applications** (Martin Kleppmann, 2017)
7. **The Transaction Concept: Virtues and Limitations** (H.T. Kung & J.T. Robinson, 1981)

### 实用工具
1. **数据库工具**：
   - MySQL Workbench（官方GUI工具）
   - RedisInsight（Redis可视化工具）
   - Kafka UI（Kafka管理界面）
   - TiDB Dashboard（TiDB监控面板）

2. **性能分析**：
   - EXPLAIN（查询执行计划）
   - SHOW PROFILE（MySQL性能分析）
   - Redis SLOWLOG（慢查询日志）
   - Kafka Burrow（消费者延迟监控）

3. **压测工具**：
   - sysbench（数据库压测）
   - redis-benchmark（Redis压测）
   - Kafka Producer/Consumer Perf（Kafka压测）
   - JMeter（全栈压测）

## 学习产出要求

### 设计产出
1. ✅ 多级缓存架构设计文档（含一致性方案）
2. ✅ MySQL主从复制与读写分离方案
3. ✅ 分布式事务协调方案（Saga/TCC模式）
4. ✅ Kafka消息可靠性保证方案
5. ✅ TiDB HTAP架构设计文档
6. ✅ 数据迁移与同步方案
7. ✅ 监控告警体系设计（SLI/SLO定义）

### 代码产出
1. ✅ 缓存一致性实现（延迟双删、消息通知）
2. ✅ 分布式锁实现（Redis Redlock）
3. ✅ Kafka Exactly Once消费者实现
4. ✅ MySQL分库分表路由算法
5. ✅ 连接池优化配置（HikariCP）
6. ✅ 数据库迁移脚本（MySQL到TiDB）
7. ✅ 监控指标采集配置

### 技能验证
1. ✅ 能够设计高性能缓存架构（多级缓存、一致性保证）
2. ✅ 精通MySQL性能优化（索引、查询、事务、锁）
3. ✅ 掌握Redis Cluster集群管理与运维
4. ✅ 能够设计Kafka消息系统（分区、副本、消费组）
5. ✅ 理解TiDB分布式架构与HTAP模式
6. ✅ 能够设计分布式事务方案（Saga、TCC、消息事务）
7. ✅ 掌握数据同步与迁移技术（CDC、DM）
8. ✅ 能够进行数据库容量规划与成本优化

### 文档产出
1. ✅ 架构决策记录（ADR）3-5篇
2. ✅ 技术选型对比文档（Kafka vs Pulsar、MySQL vs TiDB）
3. ✅ 性能测试报告（MySQL/Redis/Kafka压测）
4. ✅ 数据系统设计最佳实践总结

## 时间安排建议

### 第1天（MySQL深度解析与性能调优）
- **上午（4小时）**：MySQL架构与核心原理
  - InnoDB存储引擎、索引原理、事务与锁
  - 主从复制与高可用
  - 实践：分析慢查询与优化SQL

- **下午（4小时）**：MySQL性能调优实践
  - 查询优化、表结构优化、连接池配置
  - 分库分表设计
  - 实践：配置MySQL主从复制

- **晚上（2小时）**：MySQL监控与运维
  - Performance Schema使用
  - 备份与恢复策略
  - 实践：配置MySQL监控告警

### 第2天（Redis深度应用与缓存架构）
- **上午（4小时）**：Redis核心原理与实践
  - 数据结构实现、持久化机制
  - 主从复制、哨兵、Cluster
  - 实践：搭建Redis Cluster

- **下午（4小时）**：缓存架构设计与优化
  - 多级缓存策略、缓存一致性
  - 分布式锁实现
  - 实践：设计多级缓存方案

- **晚上（2小时）**：Redis性能优化与监控
  - 内存优化、大Key/热Key处理
  - 缓存监控告警
  - 实践：配置Redis监控

### 第3天（Kafka与Pulsar消息系统）
- **上午（4小时）**：Kafka深度解析
  - Kafka架构、消息存储、生产者/消费者
  - 消息投递语义与事务消息
  - 实践：搭建Kafka集群

- **下午（4小时）**：Pulsar架构与实践
  - Pulsar架构、存储分离、订阅模型
  - 分层存储与多地域复制
  - 实践：对比Kafka与Pulsar

- **晚上（2小时）**：消息系统监控与优化
  - 消息积压处理
  - Kafka/Pulsar监控配置
  - 实践：设计消息可靠性方案

### 第4天（TiDB分布式数据库与架构设计）
- **上午（4小时）**：TiDB架构与HTAP
  - TiDB/PD/TiKV/TiFlash架构
  - 分布式事务与数据分片
  - 实践：部署TiDB集群

- **下午（4小时）**：数据系统架构设计
  - 数据存储选型决策
  - 分布式事务方案设计
  - 实践：编写数据架构设计文档

- **晚上（2小时）**：总结与知识体系构建
  - 数据系统设计最佳实践
  - 架构决策记录编写
  - 制定后续学习计划

## 学习方法建议

### 1. 理论与实践结合（40%理论 + 60%实践）
- 阅读官方文档理解核心原理
- 使用Docker快速搭建实验环境
- 动手实践各种配置与调优
- 编写技术方案文档巩固理解

### 2. 对比学习方法
- 对比MySQL与TiDB的架构差异
- 对比Kafka与Pulsar的设计理念
- 对比不同缓存策略的适用场景
- 对比分布式事务方案的优缺点

### 3. 性能驱动学习
- 使用压测工具验证理论
- 监控指标指导优化方向
- 基准测试对比技术选型
- 性能分析驱动架构改进

### 4. 生产视角思考
- 考虑高可用与容灾方案
- 关注故障恢复与降级策略
- 重视监控与可观测性
- 思考成本优化与容量规划

### 5. 与其他路径协同
- 与分布式系统路径：理解底层一致性协议
- 与云原生路径：掌握K8s上的数据系统部署
- 与数据工程路径：学习数据处理Pipeline
- 与高性能API路径：设计高并发数据访问

## 常见问题与解决方案

### Q1：MySQL分库分表怎么做？
**A**：分库分表需要考虑：
1. **分片策略**：范围分片、哈希分片、地理位置分片
2. **分片键选择**：选择查询频率高且分布均匀的字段
3. **路由实现**：客户端路由、代理路由（MyCAT、ShardingSphere）
4. **扩容方案**：预留扩容空间、使用一致性哈希
5. **跨分片问题**：JOIN、排序、分页需要特殊处理

### Q2：缓存一致性如何保证？
**A**：常见的缓存一致性方案：
1. **Cache-Aside + 延迟双删**：更新DB后延迟删除缓存
2. **消息通知**：通过消息队列异步删除缓存
3. **Canal CDC**：监听MySQL Binlog自动更新缓存
4. **分布式锁**：更新时加锁防止并发问题
5. **短TTL**：设置较短的过期时间作为兜底

### Q3：Kafka消息积压怎么处理？
**A**：消息积压处理策略：
1. **增加消费者**：扩容消费者实例（不超过分区数）
2. **增加分区**：重新分区提高并行度
3. **丢弃非关键消息**：临时降级处理
4. **批量处理**：优化消费者批量消费
5. **扩容消费者后Rebalance**：触发重新分配分区
6. **下游消费优化**：优化消费者处理逻辑

### Q4：TiDB vs MySQL怎么选？
**A**：选型决策：
- **选MySQL**：
  - 数据量 < 1TB
  - 写入QPS < 10万
  - 强依赖复杂事务
  - 团队熟悉MySQL

- **选TiDB**：
  - 数据量 > 1TB且持续增长
  - 需要水平扩展能力
  - 需要HTAP混合负载
  - 需要强一致性且可扩展

### Q5：Redis Cluster vs Sentinel怎么选？
**A**：选型决策：
- **Sentinel**：
  - 数据量 < 10GB
  - 写入QPS < 10万
  - 只需要高可用，不需要分片

- **Cluster**：
  - 数据量 > 10GB
  - 写入QPS > 10万
  - 需要自动分片和水平扩展

### Q6：如何保证Kafka消息不丢失？
**A**：消息可靠性保证：
1. **生产端**：
   - `acks=all`（等待所有副本确认）
   - `retries > 0`（自动重试）
   - `enable.idempotence=true`（幂等性）

2. **Broker端**：
   - `replication.factor > 2`（多副本）
   - `min.insync.replicas > 1`（最少同步副本）
   - `unclean.leader.election.enable=false`（禁止非同步副本当选Leader）

3. **消费端**：
   - `enable.auto.commit=false`（手动提交）
   - 处理成功后再提交offset

### Q7：分布式事务如何实现？
**A**：分布式事务方案：
1. **Saga模式**：长事务拆分为多个本地事务 + 补偿
2. **TCC模式**：Try-Confirm-Cancel三阶段
3. **本地消息表**：本地事务 + 消息表 + 定时任务
4. **事务消息**：Kafka/RocketMQ事务消息
5. **Seata框架**：AT/TCC/SAGA模式统一实现

## 知识体系构建

### 核心知识领域

#### 1. 关系型数据库（MySQL）
```
MySQL知识体系
├── 存储引擎
│   ├── InnoDB（默认，支持事务）
│   ├── MyISAM（表级锁）
│   └── Memory（内存表）
├── 索引优化
│   ├── B+树索引（聚簇索引、二级索引）
│   ├── 哈希索引（自适应哈希）
│   ├── 全文索引（全文搜索）
│   └── 函数索引（MySQL 8.0+）
├── 事务与锁
│   ├── ACID特性
│   ├── 隔离级别（4种）
│   ├── MVCC实现
│   └── 锁机制（行锁、间隙锁、临键锁）
├── 主从复制
│   ├── binlog格式（Statement、Row、Mixed）
│   ├── 异步复制
│   ├── 半同步复制
│   └── GTID复制
└── 性能调优
    ├── 查询优化（EXPLAIN、索引覆盖）
    ├── 表结构优化（数据类型、范式）
    ├── 连接池配置（HikariCP）
    └── 监控指标（QPS、TPS、连接数）
```

#### 2. 分布式缓存（Redis）
```
Redis知识体系
├── 数据结构
│   ├── String（字符串）
│   ├── Hash（哈希表）
│   ├── List（列表）
│   ├── Set（集合）
│   ├── ZSet（有序集合）
│   ├── Bitmap（位图）
│   ├── HyperLogLog（基数统计）
│   ├── GEO（地理位置）
│   └── Stream（流）
├── 底层实现
│   ├── SDS（简单动态字符串）
│   ├── 跳表（ZSet底层）
│   ├── 压缩列表（小数据集）
│   └── 整数集合（Set底层）
├── 持久化
│   ├── RDB（快照）
│   ├── AOF（追加日志）
│   └── 混合持久化（RDB + AOF）
├── 高可用
│   ├── 主从复制
│   ├── Sentinel（哨兵）
│   └── Cluster（集群）
└── 应用模式
    ├── 缓存（Cache-Aside）
    ├── 分布式锁（Redlock）
    ├── 限流（令牌桶、滑动窗口）
    ├── 排行榜（ZSet）
    └── 计数器（String INCR）
```

#### 3. 消息队列（Kafka/Pulsar）
```
消息队列知识体系
├── Kafka架构
│   ├── Broker（代理节点）
│   ├── Topic（主题）
│   ├── Partition（分区）
│   ├── Segment（段文件）
│   ├── Producer（生产者）
│   └── Consumer（消费者）
├── Pulsar架构
│   ├── Broker（无状态计算）
│   ├── BookKeeper（有状态存储）
│   ├── Zookeeper（元数据）
│   ├── Layered Storage（分层存储）
│   └── Geo-Replication（地域复制）
├── 消息投递语义
│   ├── At Most Once（最多一次）
│   ├── At Least Once（至少一次）
│   └── Exactly Once（精确一次）
├── 消息可靠性
│   ├── 副本机制（多副本复制）
│   ├── ISR（同步副本队列）
│   ├── HW（高水位）
│   └── 事务消息
└── 应用场景
    ├── 异步解耦
    ├── 流量削峰
    ├── 数据同步（CDC）
    ├── 日志收集
    └── 事件溯源
```

#### 4. 分布式数据库（TiDB）
```
TiDB知识体系
├── 架构组件
│   ├── PD（Placement Driver，调度器）
│   ├── TiDB（计算层，SQL解析）
│   ├── TiKV（存储层，分布式KV）
│   └── TiFlash（分析层，列式存储）
├── 核心技术
│   ├── Raft协议（一致性）
│   ├── MVCC（多版本并发控制）
│   ├── PD调度（数据调度）
│   ├── 2PC（两阶段提交）
│   └── HTAP（混合事务分析）
├── 数据分布
│   ├── Region（数据分片）
│   ├── Replica（副本）
│   ├── Raft Group（Raft组）
│   └── 数据路由
└── 迁移同步
    ├── DM（Data Migration）
    ├── CDC（Change Data Capture）
    └── Binlog同步
```

### 学习深度建议

#### 精通级别
- **MySQL索引优化**：能够分析执行计划并优化复杂查询
- **Redis缓存架构**：设计多级缓存与一致性方案
- **分布式事务**：理解Saga/TCC模式并能够设计实现
- **Kafka消息可靠性**：配置 Exactly Once 语义
- **TiDB HTAP**：理解行存与列存的配合使用

#### 掌握级别
- **MySQL主从复制**：配置主从复制与故障切换
- **Redis Cluster**：搭建与运维Redis Cluster
- **Kafka集群管理**：分区管理、消费者组管理
- **Pulsar分层存储**：配置热/冷数据存储
- **数据迁移**：使用DM/Debezium同步数据

#### 了解级别
- **MySQL源码**：InnoDB实现原理
- **Redis源码**：数据结构底层实现
- **Raft协议**：分布式共识算法细节
- **其他数据库**：MongoDB、PostgreSQL、Cassandra
- **NewSQL**：CockroachDB、Spanner、Aurora

## 下一步学习

### 立即进入
1. **分布式系统复习**（路径05）：
   - 理解分布式一致性协议（Raft、Paxos）
   - 学习分布式事务模式（Saga、TCC）
   - 协同效应：本路径的技术实践 + 分布式路径的理论基础

2. **实践项目实现**：
   - 高性能API项目（docs/practice-projects/high-performance-api.md）
   - 全栈应用项目（docs/practice-projects/fullstack-application.md）

### 后续深入
1. **云原生进阶**（路径06）：K8s上的数据系统部署
2. **数据工程**（路径07）：大规模数据处理Pipeline
3. **AI/ML系统**（路径04）：向量数据库与特征存储

### 持续跟进
- **数据库技术演进**：NewSQL、Serverless数据库
- **消息系统发展**：Kafka、Pulsar新版本特性
- **缓存技术**：Dragonfly、KeyDB等Redis替代方案
- **数据湖仓**：Delta Lake、Apache Iceberg、Hudi
- **技术博客**：Percona、PingCAP、Confluent

---

## 学习路径特点

### 针对人群
- 有数据库使用经验，需要系统化深入理解
- 面向JD中的"数据存储与处理"要求
- 适合需要设计高性能数据平台的工程师

### 学习策略
- **高强度**：4天集中学习，每天8-10小时
- **重实践**：60%时间动手实践，40%理论学习
- **JD导向**：所有学习内容都对应JD要求
- **技术栈明确**：MySQL 8.0+、Redis 7.0+、Kafka 3.6+、TiDB 7.1+
- **对比学习**：理解不同技术的适用场景

### 协同学习
- 与分布式系统路径并行：理论与实践结合
- 与云原生路径协同：K8s上的数据系统部署
- 与数据工程路径互补：存储与处理相配合

### 质量保证
- 所有资源都是权威、最新（2023-2024）
- 技术栈版本明确
- 实践项目可操作（Docker Compose一键部署）
- 产出可验证（架构文档、代码示例、测试报告）

---

*学习路径设计：针对有数据库经验的后端工程师，系统化学习分布式存储与消息系统*
*时间窗口：春节第2周中间4天，高强度快速提升数据系统能力*
*JD对标：满足JD中数据库调优、分布式系统、数据处理Pipeline等核心要求*
