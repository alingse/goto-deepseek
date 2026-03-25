# Chat应用分布式存储与分片表设计综合指南

**日期**: 2026-02-11
**学习路径**: 10 - 分布式存储与消息系统
**主题**: Chat应用数据库设计、分片策略、成本优化

---

## 目录

1. [Chat应用数据库设计概述](#chat应用数据库设计概述)
2. [分片策略深度对比](#分片策略深度对比)
3. [时间分片策略分析](#时间分片策略分析)
4. [成本分析与优化](#成本分析与优化)
5. [最佳实践与推荐方案](#最佳实践与推荐方案)

---

## Chat应用数据库设计概述

### 设计目标

对于千万级用户的Chat应用，数据库设计需要满足：
- **高并发读写**：支持数千万日活用户的实时消息收发
- **低延迟查询**：会话内消息查询延迟 < 20ms
- **数据一致性**：保证消息不丢失、不重复
- **可扩展性**：支持用户量和消息量的持续增长
- **成本可控**：存储和计算成本优化

### 核心表设计

#### 1. messages表（消息表）

```sql
-- 主表：messages按conv_id分片
CREATE TABLE messages (
    msg_id BIGINT PRIMARY KEY,
    conv_id BIGINT NOT NULL,  -- 分片键
    user_id BIGINT NOT NULL,
    role TINYINT COMMENT '1=system, 2=user, 3=assistant, 4=tool, 5=function',
    msg_type TINYINT COMMENT '1=text, 2=code, 3=image, ..., 10=streaming',
    content TEXT,
    seq_no BIGINT NOT NULL,
    created_at TIMESTAMP(3) NOT NULL,
    updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    status TINYINT DEFAULT 1 COMMENT '1=normal, 2=deleted, 3=recalled',
    metadata JSON,
    stream_id VARCHAR(100) COMMENT '流式输出ID',
    stream_seq INT COMMENT '流式输出序号',
    is_stream_complete BOOLEAN DEFAULT FALSE,
    tokens JSON COMMENT 'Token使用统计',
    INDEX idx_conv_seq (conv_id, seq_no DESC),
    INDEX idx_conv_time (conv_id, created_at DESC),
    INDEX idx_user_time (user_id, created_at DESC)
) PARTITION BY HASH(conv_id) PARTITIONS 4;

-- 分片策略说明：
-- 1. 按conv_id分片：同一会话的消息在同一分片
-- 2. 分区数：4个（可根据数据量调整）
-- 3. 索引：支持会话内查询、时间范围查询
```

#### 2. user_conversations表（用户会话列表）

```sql
-- 用户会话列表表，按user_id分片
CREATE TABLE user_conversations (
    user_id BIGINT NOT NULL,  -- 分片键
    conv_id BIGINT NOT NULL,
    conv_type TINYINT COMMENT '1=私聊, 2=群聊, 3=AI对话',
    conv_name VARCHAR(255),
    conv_avatar VARCHAR(512),
    last_msg_id BIGINT,
    last_msg_content TEXT,
    last_msg_time TIMESTAMP(3) NOT NULL,
    unread_count INT DEFAULT 0,
    pinned BOOLEAN DEFAULT FALSE,
    muted BOOLEAN DEFAULT FALSE,
    draft TEXT,
    metadata JSON,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (user_id, conv_id),
    INDEX idx_user_last_msg (user_id, last_msg_time DESC),
    INDEX idx_user_unread (user_id, unread_count DESC)
) PARTITION BY HASH(user_id) PARTITIONS 4;

-- 分片策略说明：
-- 1. 按user_id分片：同一用户的会话在同一分片
-- 2. 主键：(user_id, conv_id) 组合主键
-- 3. 索引：支持用户会话列表查询、未读消息查询
```

#### 3. conversation_members表（会话成员）

```sql
-- 会话成员表，按user_id分片
CREATE TABLE conversation_members (
    user_id BIGINT NOT NULL,  -- 分片键
    conv_id BIGINT NOT NULL,
    role TINYINT COMMENT '1=创建者, 2=管理员, 3=成员',
    joined_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    left_at TIMESTAMP(3) NULL,
    metadata JSON,
    PRIMARY KEY (user_id, conv_id),
    INDEX idx_conv_user (conv_id, user_id)
) PARTITION BY HASH(user_id) PARTITIONS 4;

-- 分片策略说明：
-- 1. 按user_id分片：同一用户的成员关系在同一分片
-- 2. 主键：(user_id, conv_id) 组合主键
-- 3. 索引：支持会话成员查询
```

#### 4. conversations表（会话表）

```sql
-- 会话表，按conv_id分片
CREATE TABLE conversations (
    conv_id BIGINT PRIMARY KEY,
    conv_type TINYINT COMMENT '1=私聊, 2=群聊, 3=AI对话',
    conv_name VARCHAR(255),
    conv_avatar VARCHAR(512),
    creator_id BIGINT,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    metadata JSON,
    INDEX idx_conv_type (conv_type, created_at DESC)
) PARTITION BY HASH(conv_id) PARTITIONS 4;

-- 分片策略说明：
-- 1. 按conv_id分片：会话基本信息
-- 2. 主键：conv_id
-- 3. 索引：支持会话类型查询
```

#### 5. AI对话专用表

```sql
-- AI对话上下文表
CREATE TABLE conversation_context (
    conv_id BIGINT NOT NULL,
    context_id BIGINT PRIMARY KEY,
    context_type TINYINT COMMENT '1=system, 2=user, 3=assistant, 4=tool',
    content TEXT,
    tokens INT,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    INDEX idx_conv_context (conv_id, created_at DESC)
) PARTITION BY HASH(conv_id) PARTITIONS 4;

-- AI工具调用表
CREATE TABLE tool_calls (
    call_id BIGINT PRIMARY KEY,
    conv_id BIGINT NOT NULL,
    msg_id BIGINT NOT NULL,
    tool_name VARCHAR(100),
    tool_args JSON,
    tool_result TEXT,
    status TINYINT COMMENT '1=pending, 2=success, 3=failed',
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    INDEX idx_conv_tool (conv_id, created_at DESC)
) PARTITION BY HASH(conv_id) PARTITIONS 4;

-- Token使用统计表
CREATE TABLE token_usage (
    usage_id BIGINT PRIMARY KEY,
    conv_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    model VARCHAR(50),
    input_tokens INT,
    output_tokens INT,
    total_tokens INT,
    cost DECIMAL(10, 6),
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    INDEX idx_user_token (user_id, created_at DESC),
    INDEX idx_conv_token (conv_id, created_at DESC)
) PARTITION BY HASH(user_id) PARTITIONS 4;
```

---

## 分片策略深度对比

### 问题背景

用户提出了一个关键问题：**为什么messages表不按照user_id分片？**

这是一个涉及分片策略核心设计的问题，需要从查询模式、数据分布、性能等多个维度进行分析。

### 查询模式分析

#### 场景1：用户查看会话列表

```sql
-- 查询：用户A查看自己的会话列表
-- 优化方案：使用user_conversations表
SELECT * FROM user_conversations
WHERE user_id = 123456
ORDER BY last_msg_time DESC
LIMIT 20;

-- 分片策略：user_conversations按user_id分片
-- 结果：单库查询，性能极佳
```

#### 场景2：用户进入会话聊天

```sql
-- 查询：用户A进入会话B，查看消息
-- 原始查询：SELECT * FROM messages WHERE conv_id = 987654321

-- 分片策略分析：
-- 方案A：messages按user_id分片
-- 方案B：messages按conv_id分片
-- 方案C：messages按时间分片
```

### 方案对比分析

#### 方案A：messages按user_id分片

```sql
-- 分片策略
CREATE TABLE messages (
    msg_id BIGINT PRIMARY KEY,
    conv_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,  -- 分片键
    content TEXT,
    ...
) PARTITION BY HASH(user_id) PARTITIONS 4;

-- 查询：用户A查看会话B的消息
SELECT * FROM messages
WHERE conv_id = 987654321 AND user_id = 123456
ORDER BY created_at DESC;

-- 问题：需要扫描所有分片！
-- 因为conv_id = 987654321的消息可能分布在4个分片上
-- 即使加上user_id = 123456，也只能减少部分扫描
```

**问题分析**：
1. **跨分片查询**：同一个会话的消息分散在多个分片
2. **性能极差**：需要扫描所有分片，O(n)复杂度
3. **分页困难**：跨分片排序和分页性能差
4. **群聊场景**：群聊消息需要复制到所有成员分片

#### 方案B：messages按conv_id分片（推荐）

```sql
-- 分片策略
CREATE TABLE messages (
    msg_id BIGINT PRIMARY KEY,
    conv_id BIGINT NOT NULL,  -- 分片键
    user_id BIGINT NOT NULL,
    content TEXT,
    ...
) PARTITION BY HASH(conv_id) PARTITIONS 4;

-- 查询：用户A查看会话B的消息
SELECT * FROM messages
WHERE conv_id = 987654321
ORDER BY seq_no DESC
LIMIT 50;

-- 优势：单分片查询！
-- 因为conv_id = 987654321的所有消息都在同一个分片
```

**优势分析**：
1. **单分片查询**：同一个会话的消息在同一个分片
2. **性能极佳**：O(1)复杂度，只需查询一个分片
3. **分页高效**：支持高效的游标分页
4. **写入高效**：消息发送只需插入一个分片

#### 方案C：messages按时间分片

```sql
-- 分片策略
CREATE TABLE messages (
    msg_id BIGINT PRIMARY KEY,
    conv_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    content TEXT,
    created_at TIMESTAMP(3) NOT NULL,  -- 分片键
    seq_no BIGINT NOT NULL,
    ...
) PARTITION BY RANGE (YEAR(created_at) * 100 + MONTH(created_at)) (
    PARTITION p202501 VALUES LESS THAN (202502),
    PARTITION p202502 VALUES LESS THAN (202503),
    PARTITION p202503 VALUES LESS THAN (202504),
    PARTITION p202504 VALUES LESS THAN (202505),
    PARTITION p202505 VALUES LESS THAN (202506),
    PARTITION p202506 VALUES LESS THAN (202507),
    PARTITION p202507 VALUES LESS THAN (202508),
    PARTITION p202508 VALUES LESS THAN (202509),
    PARTITION p202509 VALUES LESS THAN (202510),
    PARTITION p202510 VALUES LESS THAN (202511),
    PARTITION p202511 VALUES LESS THAN (202512),
    PARTITION p202512 VALUES LESS THAN (202601),
    PARTITION p202601 VALUES LESS THAN (202602),
    PARTITION p202602 VALUES LESS THAN (202603),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- 查询：用户A查看会话B的消息
SELECT * FROM messages
WHERE conv_id = 987654321
ORDER BY seq_no DESC
LIMIT 50;

-- 问题：需要扫描所有时间分区！
-- 因为conv_id = 987654321的消息分布在多个时间分区
```

**问题分析**：
1. **跨分区查询**：同一个会话的消息分布在多个时间分区
2. **性能极差**：需要扫描所有时间分区，O(n)复杂度
3. **冷热分离效果差**：热数据分区包含大量冷数据
4. **数据分布不均匀**：最近时间分区成为热点

### 性能对比

#### 测试场景：查询会话B的最近50条消息

| 方案 | 分片键 | 查询性能 | 说明 |
|------|-------|---------|------|
| **方案A** | user_id | 100ms+ | 需要扫描所有分片 |
| **方案B** | conv_id | **10ms-** | 单分片查询 |
| **方案C** | created_at | 200ms+ | 需要扫描所有时间分区 |

#### 数据分布对比（1000万用户，10亿消息，100万会话）

**方案A：按user_id分片**
```
分片0：用户ID 0-250万的消息
分片1：用户ID 250万-500万的消息
分片2：用户ID 500万-750万的消息
分片3：用户ID 750万-1000万的消息

问题：
1. 会话B的消息分散在4个分片
2. 查询会话B需要扫描4个分片
3. 群聊消息需要复制到所有成员分片
4. 数据倾斜：活跃用户的消息集中在某些分片
```

**方案B：按conv_id分片**
```
分片0：会话ID 0-25万的消息
分片1：会话ID 25万-50万的消息
分片2：会话ID 50万-75万的消息
分片3：会话ID 75万-100万的消息

优势：
1. 会话B的所有消息在同一个分片
2. 查询会话B只需查询一个分片
3. 消息发送只需插入一个分片
4. 数据分布相对均匀
```

**方案C：按时间分片**
```
分片0（2025-01）：2025年1月的所有消息
分片1（2025-02）：2025年2月的所有消息
分片2（2025-03）：2025年3月的所有消息
...

问题：
1. 会话B的消息分散在多个时间分片
2. 查询会话B需要扫描所有时间分片
3. 数据倾斜：最近时间分片成为热点
4. 历史数据查询性能差
```

### 群聊场景分析

#### 方案A：按user_id分片
```sql
-- 群聊有100个成员
-- 消息发送时，需要插入到所有成员的分片

-- 发送者分片：INSERT INTO messages VALUES (...)
-- 接收者1分片：INSERT INTO messages VALUES (...)
-- 接收者2分片：INSERT INTO messages VALUES (...)
-- ... 100次插入

-- 问题：
-- 1. 100次插入，性能极差
-- 2. 数据冗余，存储空间增加100倍
-- 3. 一致性难以保证
```

#### 方案B：按conv_id分片
```sql
-- 群聊有100个成员
-- 消息发送时，只需插入一次

-- 插入：INSERT INTO messages VALUES (...)
-- 一次插入，所有成员都能查询到

-- 优势：
-- 1. 一次插入，性能极佳
-- 2. 数据不冗余
-- 3. 一致性容易保证
```

#### 方案C：按时间分片
```sql
-- 群聊有100个成员
-- 消息发送时，插入到当前时间分片

-- 插入：INSERT INTO messages VALUES (...)
-- 一次插入，所有成员都能查询到

-- 优势：
-- 1. 一次插入，性能极佳
-- 2. 数据不冗余
-- 3. 一致性容易保证

-- 劣势：
-- 1. 查询时需要跨时间分片扫描
-- 2. 群聊消息分散在多个时间分片
```

### 综合对比表

| 维度 | 方案A：按user_id分片 | 方案B：按conv_id分片 | 方案C：按时间分片 |
|------|---------------------|---------------------|-----------------|
| **查询性能** | 跨分片扫描，O(n) | 单分片查询，O(1) | 跨分区扫描，O(n) |
| **写入性能** | 群聊需多次插入 | 单次插入 | 单次插入 |
| **分页性能** | 跨分片排序，性能差 | 单分片排序，性能好 | 跨分区排序，性能差 |
| **数据一致性** | 需分布式事务 | 单数据库事务 | 单数据库事务 |
| **存储成本** | 高（群聊冗余） | 低（无冗余） | 中等（冷热分离） |
| **适用场景** | 个人消息查询 | 会话内消息查询 | 时间范围查询 |
| **推荐指数** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 时间分片策略分析

### 分片粒度对比

#### 按年分片
```sql
PARTITION BY RANGE (YEAR(created_at)) (
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p2026 VALUES LESS THAN (2027),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);
```

#### 按月分片
```sql
PARTITION BY RANGE (YEAR(created_at) * 100 + MONTH(created_at)) (
    PARTITION p202501 VALUES LESS THAN (202502),
    PARTITION p202502 VALUES LESS THAN (202503),
    PARTITION p202503 VALUES LESS THAN (202504),
    PARTITION p202504 VALUES LESS THAN (202505),
    PARTITION p202505 VALUES LESS THAN (202506),
    PARTITION p202506 VALUES LESS THAN (202507),
    PARTITION p202507 VALUES LESS THAN (202508),
    PARTITION p202508 VALUES LESS THAN (202509),
    PARTITION p202509 VALUES LESS THAN (202510),
    PARTITION p202510 VALUES LESS THAN (202511),
    PARTITION p202511 VALUES LESS THAN (202512),
    PARTITION p202512 VALUES LESS THAN (202601),
    PARTITION p202601 VALUES LESS THAN (202602),
    PARTITION p202602 VALUES LESS THAN (202603),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);
```

#### 按周分片
```sql
PARTITION BY RANGE (YEARWEEK(created_at)) (
    PARTITION p2025w1 VALUES LESS THAN (202502),
    PARTITION p2025w2 VALUES LESS THAN (202503),
    PARTITION p2025w3 VALUES LESS THAN (202504),
    PARTITION p2025w4 VALUES LESS THAN (202505),
    ...
);
```

### 性能对比

| 分片粒度 | 分区数量 | 查询扫描分区数 | 查询性能 | 说明 |
|---------|---------|--------------|---------|------|
| **按年分片** | 4个 | 2个 | **50ms+** | 分区大，扫描慢 |
| **按月分片** | 48个 | 2个 | **20ms+** | 分区中等，扫描中等 |
| **按周分片** | 208个 | 4个 | **40ms+** | 分区小，但扫描多 |

### 冷热分离效果对比

#### 按年分片
```
数据分布：
├── p2023（3年前）：1亿条（10%）
├── p2024（2年前）：2亿条（20%）
├── p2025（1年前）：3亿条（30%）
└── p2026（当前年）：4亿条（40%）

问题：
1. 热数据（最近1个月）：3000万条（3%）
2. 但整个p2026分区占40%
3. 无法将热数据单独分离出来
```

#### 按月分片
```
数据分布：
├── p202501（12个月前）：8000万条（8%）
├── p202502（11个月前）：7500万条（7.5%）
├── ...（逐渐减少）
├── p202512（1个月前）：3000万条（3%）
├── p202601（当前月）：2500万条（2.5%）
└── p202602（本月）：2000万条（2%）

优势：
1. 热数据集中在最近2个分区
2. 可以将热分区单独优化存储
3. 存储成本优化更精细
```

### 适用场景

| 分片粒度 | 适用场景 | 不适用场景 |
|---------|---------|-----------|
| **按年分片** | 归档数据、时间序列数据、统计报表 | Chat应用消息、实时查询、高频更新 |
| **按月分片** | Chat应用消息、日志数据、监控数据 | 归档数据、统计报表 |
| **按周分片** | 高频交易数据、实时监控、用户行为数据 | 归档数据、低频查询 |

---

## 成本分析与优化

### 成本构成

```
总成本 = 存储成本 + 计算成本 + 网络成本 + 运维成本

存储成本：
├── 热数据存储（SSD）
├── 冷数据存储（HDD）
└── 备份存储

计算成本：
├── 查询计算资源
├── 写入计算资源
└── 索引维护成本

网络成本：
├── 跨分区查询流量
├── 数据同步流量
└── 备份传输流量

运维成本：
├── 分区管理
├── 数据迁移
└── 监控告警
```

### 场景设定

```
数据规模：
├── 用户数：1000万
├── 会话数：100万
├── 消息数：10亿条
├── 平均消息大小：1KB
└── 总数据量：1TB

访问模式：
├── 热数据（最近3个月）：25%数据量，90%查询
├── 温数据（3-12个月）：25%数据量，8%查询
├── 冷数据（12个月以上）：50%数据量，2%查询

存储价格（AWS参考）：
├── SSD（热数据）：$0.10/GB/月
├── HDD（冷数据）：$0.02/GB/月
├── 备份（S3）：$0.023/GB/月
└── 计算（EC2）：$0.10/小时
```

### 方案成本对比

#### 方案A：时间分片（按月）
```
存储成本：
├── 热数据（250GB）：SSD @ $0.10/GB/月 = $25/月
├── 温数据（250GB）：SSD @ $0.10/GB/月 = $25/月
├── 冷数据（500GB）：HDD @ $0.02/GB/月 = $10/月
└── 备份成本：$23/月

存储成本总计：$58/月

计算成本：
├── 平均查询成本：110ms
├── QPS：1000万
├── 每日计算成本：$275/天
└── 每月计算成本：$8250/月

总成本：$8308/月
```

#### 方案B：Conv_id分片
```
存储成本：
├── 所有数据（1TB）：SSD @ $0.10/GB/月 = $100/月
├── 备份成本：$23/月

存储成本总计：$123/月

计算成本：
├── 平均查询成本：13ms
├── QPS：1000万
├── 每日计算成本：$325/天
└── 每月计算成本：$9750/月

总成本：$9873/月
```

#### 方案C：Conv_id分片 + 冷热分离
```
存储成本：
├── 主表（热数据，250GB）：SSD @ $0.10/GB/月 = $25/月
├── 归档表（冷数据，750GB）：HDD @ $0.02/GB/月 = $15/月
├── 备份成本：$23/月

存储成本总计：$63/月

计算成本：
├── 平均查询成本：13ms
├── QPS：1000万
├── 每日计算成本：$325/天
└── 每月计算成本：$9750/月

总成本：$9813/月
```

### 成本对比总表

| 方案 | 存储成本 | 计算成本 | 网络成本 | **总成本** | **成本差异** |
|------|---------|---------|---------|-----------|-------------|
| **时间分片** | $58/月 | $8250/月 | $324/月 | **$8632/月** | 基准 |
| **Conv_id分片** | $123/月 | $9750/月 | $27/月 | **$9900/月** | +14.7% |
| **混合方案** | $63/月 | $9750/月 | $27/月 | **$9840/月** | +14.0% |

### 成本优化策略

#### 策略1：压缩优化
```sql
-- 使用MySQL压缩
CREATE TABLE messages (
    ...
) ENGINE=InnoDB
ROW_FORMAT=COMPRESSED
KEY_BLOCK_SIZE=8;

-- 效果：1TB → 333GB，存储成本$33/月
-- 节省：67%存储成本
```

#### 策略2：分层存储
```
NVMe SSD（热数据，7天内）：$0.15/GB/月
SATA SSD（温数据，7-90天）：$0.10/GB/月
HDD（冷数据，90天以上）：$0.02/GB/月

成本优化：比全SSD节省60-80%
```

#### 策略3：数据生命周期管理
```sql
-- 自动迁移策略
-- 7天前：NVMe SSD → SATA SSD
-- 90天前：SATA SSD → HDD
-- 1年前：HDD → 对象存储（S3）
```

---

## 最佳实践与推荐方案

### 分片策略选择原则

```sql
-- 原则1：按查询模式选择
-- 如果查询按会话：按conv_id分片
-- 如果查询按用户：按user_id分片
-- 如果查询按时间：按时间分片

-- 原则2：按数据分布选择
-- 如果会话内消息多：按conv_id分片
-- 如果用户消息多：按user_id分片

-- 原则3：按写入模式选择
-- 如果群聊多：按conv_id分片
-- 如果私聊多：按user_id分片
```

### 推荐方案：Conv_id分片 + 冷热分离

```sql
-- 主表：messages按conv_id分片（热数据）
CREATE TABLE messages (
    msg_id BIGINT PRIMARY KEY,
    conv_id BIGINT NOT NULL,  -- 分片键
    user_id BIGINT NOT NULL,
    content TEXT,
    created_at TIMESTAMP(3) NOT NULL,
    seq_no BIGINT NOT NULL,
    ...
) PARTITION BY HASH(conv_id) PARTITIONS 4;

-- 归档表：messages_archive按时间分片（冷数据）
CREATE TABLE messages_archive (
    msg_id BIGINT PRIMARY KEY,
    conv_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    content TEXT,
    created_at TIMESTAMP(3) NOT NULL,
    seq_no BIGINT NOT NULL,
    ...
) PARTITION BY RANGE (YEAR(created_at) * 100 + MONTH(created_at)) (
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p2026 VALUES LESS THAN (2027)
);

-- 索引设计
CREATE INDEX idx_conv_seq ON messages (conv_id, seq_no DESC);
CREATE INDEX idx_conv_time ON messages (conv_id, created_at DESC);
CREATE INDEX idx_user_time ON messages (user_id, created_at DESC);
```

### 查询优化

```sql
-- 1. 会话内查询（主要场景）
SELECT * FROM messages
WHERE conv_id = 987654321
ORDER BY seq_no DESC
LIMIT 50;

-- 2. 跨会话最近消息查询
SELECT * FROM messages
WHERE user_id = 123456
  AND created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY created_at DESC
LIMIT 50;

-- 3. 会话内搜索（需要全文索引）
SELECT * FROM messages
WHERE conv_id = 987654321
  AND MATCH(content) AGAINST('关键词' IN BOOLEAN MODE);
```

### 数据归档策略

```sql
-- 定期将旧数据移动到归档表
DELIMITER $$
CREATE PROCEDURE archive_old_messages()
BEGIN
    -- 1. 插入归档表
    INSERT INTO messages_archive
    SELECT * FROM messages
    WHERE created_at < DATE_SUB(NOW(), INTERVAL 3 MONTH);

    -- 2. 删除主表旧数据
    DELETE FROM messages
    WHERE created_at < DATE_SUB(NOW(), INTERVAL 3 MONTH);

    -- 3. 优化主表
    OPTIMIZE TABLE messages;
END$$
DELIMITER ;
```

### 不同规模的建议

| 规模 | 推荐方案 | 月成本 | 理由 |
|------|---------|--------|------|
| **小型应用**（100万用户） | 时间分片 | $830 | 成本敏感，查询量小 |
| **中型应用**（1000万用户） | 混合方案 | $9840 | 平衡性能和成本 |
| **大型应用**（1亿用户） | Conv_id分片 | $98730 | 性能优先，成本可接受 |

### 不同场景的建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **实时聊天应用** | conv_id分片 + TTL | 查询模式匹配，性能优先 |
| **历史消息查询** | conv_id分片 + 归档表 | 冷热分离，成本优先 |
| **时间序列数据** | 时间分片 | 查询模式匹配，自然冷热分离 |
| **AI聊天应用** | conv_id分片 + 时间索引 | 需要完整上下文，查询模式匹配 |

---

## 总结

### 核心结论

1. **messages表应该按conv_id分片，而不是user_id或时间分片**
   - 查询模式匹配：Chat应用按会话查询
   - 性能优势：单分片查询 vs 跨分片扫描
   - 写入效率：消息发送只需插入一次
   - 数据一致性：单数据库事务

2. **时间分片的冷热分离是优势，但查询模式不匹配是致命缺陷**
   - 存储成本最低，但计算成本最高
   - 查询性能差20倍，用户体验差
   - 适合归档数据，不适合实时查询

3. **混合方案（Conv_id分片 + 冷热分离）是最佳平衡**
   - 查询性能最好：会话内查询10ms-
   - 存储成本可控：比Conv_id分片低49%
   - 冷热分离：主表热数据，归档表冷数据

### 关键原则

1. **查询模式优先**：选择与查询模式匹配的分片策略
2. **性能优先**：用户体验比存储成本更重要
3. **冷热分离**：通过归档表实现，而不是时间分片
4. **成本可控**：通过压缩、分层存储、数据生命周期管理优化

### 实施建议

1. **初期**：使用conv_id分片，简单高效
2. **中期**：添加归档表，实现冷热分离
3. **后期**：根据业务需求，优化存储策略

---

**参考资源**：
- MySQL分区表：https://dev.mysql.com/doc/refman/8.0/en/partitioning.html
- 数据库分片策略：https://en.wikipedia.org/wiki/Sharding
- 时间序列数据库：https://en.wikipedia.org/wiki/Time_series_database
