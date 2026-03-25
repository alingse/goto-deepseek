# MySQL深度学习：经典机制与现代演进

**日期**: 2026-02-11
**学习路径**: 05 - 分布式系统复习
**对话主题**: MySQL经典架构、核心机制与近年来的更新迭代

## 问题背景

作为资深分布式工程师，需要深入理解MySQL在高并发、大规模数据处理场景下的核心机制。从`fullstack-jd.md`中的职责来看，需要处理：
1. **高并发服务端**：数千万日活用户的数据库设计与调优
2. **大规模数据处理**：数据采集、清洗、索引系统的数据库支撑
3. **分布式系统**：分库分表、分布式事务、高可用架构

本笔记聚焦MySQL的经典机制（InnoDB、事务、锁、索引）和近年来（MySQL 8.0+）的重要更新迭代。

## 核心知识点

### 一、MySQL经典架构与核心机制

#### 1.1 InnoDB存储引擎核心架构

**经典架构（至今仍是核心）**：
```
┌─────────────────────────────────────────────────┐
│                   MySQL Server                   │
├─────────────────────────────────────────────────┤
│              InnoDB Storage Engine              │
├─────────────────────────────────────────────────┤
│  Buffer Pool  │  Change Buffer  │  Adaptive Hash │
├─────────────────────────────────────────────────┤
│  Redo Log Buffer  │  Doublewrite Buffer         │
├─────────────────────────────────────────────────┤
│  Undo Log (Rollback Segments)                   │
└─────────────────────────────────────────────────┘
```

**关键机制**：
- **Buffer Pool**：缓存数据页和索引页，LRU算法管理
- **Change Buffer**：非唯一索引的写入缓冲，合并写入
- **Doublewrite Buffer**：防止页断裂，保证数据持久性
- **Redo Log**：WAL（Write-Ahead Logging）机制，保证崩溃恢复
- **Undo Log**：MVCC多版本并发控制的基础

#### 1.2 MVCC（多版本并发控制）机制

**核心原理**：
```sql
-- 每个事务开始时生成Read View
-- 包含：活跃事务ID列表、最小/最大事务ID、创建事务ID
-- 通过版本链判断数据可见性

-- 版本链结构（隐藏字段）：
-- DB_TRX_ID (6字节)：最后修改的事务ID
-- DB_ROLL_PTR (7字节)：回滚指针，指向Undo Log
-- DB_ROW_ID (6字行ID)：自增主键（如果没有显式主键）
```

**事务隔离级别实现**：
- **READ UNCOMMITTED**：读未提交，可能读到脏数据
- **READ COMMITTED**：读已提交，每次读取最新已提交版本
- **REPEATABLE READ**（InnoDB默认）：可重复读，基于Read View一致性视图
- **SERIALIZABLE**：串行化，加锁保证

**Read View生成时机**：
```sql
-- 快照读（普通SELECT）：在第一次读取时生成
-- 当前读（SELECT FOR UPDATE/LOCK IN SHARE MODE）：每次执行时生成
```

#### 1.3 锁机制

**锁的分类**：
```
锁类型：
├── 共享锁（S锁）：读锁，兼容其他S锁，不兼容X锁
├── 排他锁（X锁）：写锁，不兼容任何锁
├── 意向锁（IS/IX）：表级锁，快速判断行锁冲突
└── 间隙锁（Gap Lock）：锁定索引间隙，防止幻读

锁粒度：
├── 行锁（Row Lock）：InnoDB默认，基于索引
├── 页锁（Page Lock）：介于行锁和表锁之间
└── 表锁（Table Lock）：MyISAM默认，InnoDB也有
```

**锁升级与死锁**：
```sql
-- 锁升级场景
1. 行锁数量过多 → 升级为表锁
2. 索引失效 → 全表扫描 → 表锁
3. 大事务持有大量行锁

-- 死锁检测与处理
SHOW ENGINE INNODB STATUS;  -- 查看死锁信息
SET innodb_deadlock_detect = ON;  -- 死锁检测（默认开启）
SET innodb_lock_wait_timeout = 50;  -- 锁等待超时
```

#### 1.4 索引机制

**B+树索引结构**：
```
        [50]
       /    \
    [20,30] [60,70]
    /  |  \   /  |  \
 [10][20][30][50][60][70]
```

**索引类型**：
- **主键索引（Clustered Index）**：数据存储在叶子节点
- **二级索引（Secondary Index）**：叶子节点存储主键值
- **覆盖索引（Covering Index）**：索引包含查询所需所有字段
- **前缀索引**：对长文本字段的前N个字符建索引
- **联合索引**：多列组合索引，遵循最左前缀原则

**索引优化策略**：
```sql
-- 1. 索引下推（ICP - Index Condition Pushdown）
-- MySQL 5.6+ 默认开启
-- 将WHERE条件中的索引过滤下推到存储引擎层
EXPLAIN SELECT * FROM users WHERE name LIKE 'A%' AND age > 20;

-- 2. 索引条件下推（ICP）
-- 对于联合索引，减少回表次数
CREATE INDEX idx_name_age ON users(name, age);
-- 查询：WHERE name = 'Alice' AND age > 25
-- 索引直接过滤age，减少回表

-- 3. 索引合并（Index Merge）
-- MySQL 5.0+ 支持，对多个索引进行合并
EXPLAIN SELECT * FROM users WHERE name = 'Alice' OR age = 25;
```

### 二、MySQL 8.0+ 重要更新迭代

#### 2.1 MySQL 8.0 核心新特性

**1. 窗口函数（Window Functions）**
```sql
-- 传统分组 vs 窗口函数
-- 传统：GROUP BY会聚合，丢失明细
-- 窗口：保留明细，同时计算聚合

-- 语法结构
SELECT
    column1,
    column2,
    SUM(column3) OVER (PARTITION BY column1 ORDER BY column2) as running_total
FROM table_name;

-- 常用窗口函数
-- 1. 排名函数
ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rank
RANK() OVER (PARTITION BY dept ORDER BY salary DESC) as rank
DENSE_RANK() OVER (PARTITION BY dept ORDER BY salary DESC) as dense_rank

-- 2. 分析函数
LEAD(column, offset) OVER (...) as next_value
LAG(column, offset) OVER (...) as prev_value
FIRST_VALUE(column) OVER (PARTITION BY dept ORDER BY hire_date) as first_hire

-- 3. 聚合窗口函数
SUM(salary) OVER (PARTITION BY dept ORDER BY hire_date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) as rolling_sum
```

**2. 公用表表达式（CTE）**
```sql
-- 递归CTE（用于树形结构查询）
WITH RECURSIVE org_tree AS (
    SELECT id, name, parent_id, 1 as level
    FROM departments
    WHERE parent_id IS NULL

    UNION ALL

    SELECT d.id, d.name, d.parent_id, ot.level + 1
    FROM departments d
    JOIN org_tree ot ON d.parent_id = ot.id
)
SELECT * FROM org_tree ORDER BY level;

-- 非递归CTE（简化复杂查询）
WITH sales_summary AS (
    SELECT
        product_id,
        SUM(quantity) as total_qty,
        SUM(amount) as total_amount
    FROM sales
    WHERE sale_date >= '2024-01-01'
    GROUP BY product_id
)
SELECT
    p.name,
    s.total_qty,
    s.total_amount,
    s.total_amount / s.total_qty as avg_price
FROM sales_summary s
JOIN products p ON s.product_id = p.id
WHERE s.total_amount > 10000;
```

**3. JSON增强**
```sql
-- JSON数据类型（MySQL 5.7引入，8.0增强）
CREATE TABLE users (
    id INT PRIMARY KEY,
    profile JSON
);

-- JSON函数
INSERT INTO users VALUES (1, '{"name": "Alice", "age": 30, "tags": ["engineer", "python"]}');

-- 查询JSON字段
SELECT
    id,
    JSON_EXTRACT(profile, '$.name') as name,
    JSON_EXTRACT(profile, '$.tags[0]') as first_tag,
    JSON_LENGTH(profile, '$.tags') as tag_count
FROM users;

-- JSON路径表达式
SELECT * FROM users WHERE JSON_EXTRACT(profile, '$.age') > 25;

-- JSON数组操作
UPDATE users SET profile = JSON_ARRAY_APPEND(profile, '$.tags', 'mysql');

-- JSON索引（MySQL 8.0.17+）
CREATE INDEX idx_profile_age ON users((CAST(JSON_EXTRACT(profile, '$.age') AS UNSIGNED)));
```

**4. 原子DDL（Atomic DDL）**
```sql
-- MySQL 8.0+ 支持原子DDL操作
-- DDL操作要么完全成功，要么完全回滚

-- 示例：创建表（原子操作）
CREATE TABLE new_table (
    id INT PRIMARY KEY,
    name VARCHAR(100)
) ENGINE=InnoDB;

-- 如果创建失败，不会留下部分创建的表
-- 旧版本可能留下临时表或部分元数据

-- 支持原子操作的DDL：
-- CREATE/DROP TABLE
-- CREATE/DROP INDEX
-- ALTER TABLE（部分操作）
-- TRUNCATE TABLE
```

**5. 降序索引（Descending Indexes）**
```sql
-- MySQL 8.0+ 支持真正的降序索引
-- 旧版本：索引总是升序存储，降序查询需要额外排序

-- 创建降序索引
CREATE INDEX idx_desc ON orders (order_date DESC);

-- 联合索引支持混合排序
CREATE INDEX idx_mixed ON orders (customer_id ASC, order_date DESC);

-- 查询优化
SELECT * FROM orders
WHERE customer_id = 123
ORDER BY order_date DESC;
-- 直接使用降序索引，避免额外排序
```

**6. 直方图统计信息**
```sql
-- MySQL 8.0+ 支持直方图统计
-- 用于非索引列的基数估计

-- 创建直方图
ANALYZE TABLE users UPDATE HISTOGRAM ON age, salary WITH 1024 BUCKETS;

-- 查看直方图
SELECT * FROM information_schema.column_statistics
WHERE table_name = 'users';

-- 直方图类型
-- SINGLETON：单值直方图
-- EQUIHEIGHT：等高直方图（默认）
```

**7. 通用表空间（General Tablespaces）**
```sql
-- MySQL 8.0+ 支持通用表空间
-- 更好的空间管理和性能优化

-- 创建通用表空间
CREATE TABLESPACE ts1 ADD DATAFILE 'ts1.ibd' ENGINE=InnoDB;

-- 将表分配到通用表空间
CREATE TABLE t1 (id INT) TABLESPACE ts1;

-- 管理表空间
ALTER TABLESPACE ts1 ADD DATAFILE 'ts1_2.ibd';
ALTER TABLESPACE ts1 DROP DATAFILE 'ts1_2.ibd';
```

#### 2.2 MySQL 8.1+ 新特性

**1. 增强的JSON函数**
```sql
-- MySQL 8.0.17+ 新增JSON函数
JSON_TABLE()：将JSON数据转换为关系表格式
JSON_VALUE()：提取JSON值并转换为指定类型
JSON_SCHEMA_VALIDATION()：验证JSON是否符合JSON Schema

-- 示例：JSON_TABLE
SELECT * FROM JSON_TABLE(
    '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]',
    '$[*]' COLUMNS(
        id INT PATH '$.id',
        name VARCHAR(100) PATH '$.name'
    )
) AS jt;
```

**2. 递归CTE优化**
```sql
-- MySQL 8.0.22+ 对递归CTE的优化
-- 支持MATERIALIZED和NON_MATERIALIZED提示

WITH RECURSIVE cte AS (
    SELECT 1 as n
    UNION ALL
    SELECT n + 1 FROM cte WHERE n < 100000
)
SELECT /*+ MATERIALIZED */ * FROM cte;
-- MATERIALIZED：物化中间结果，减少内存使用
-- NON_MATERIALIZED：不物化，可能更快但内存消耗大
```

**3. 增强的窗口函数**
```sql
-- MySQL 8.0.28+ 新增窗口函数
-- PERCENT_RANK()：百分比排名
-- CUME_DIST()：累积分布
-- NTILE(n)：将数据分桶

SELECT
    employee_id,
    salary,
    PERCENT_RANK() OVER (ORDER BY salary DESC) as pct_rank,
    CUME_DIST() OVER (ORDER BY salary DESC) as cum_dist,
    NTILE(4) OVER (ORDER BY salary DESC) as quartile
FROM employees;
```

### 三、分布式场景下的MySQL应用

#### 3.1 分库分表策略

**1. 垂直拆分（Vertical Sharding）**
```sql
-- 按业务模块拆分
-- 用户库：users, user_profiles, user_auth
-- 订单库：orders, order_items, order_payments
-- 商品库：products, product_categories, product_inventory

-- 优点：业务清晰，减少单库压力
-- 缺点：跨库JOIN困难，需要应用层处理
```

**2. 水平拆分（Horizontal Sharding）**
```sql
-- 按数据范围或哈希分片
-- 分片策略：
-- 1. 范围分片：按ID范围（1-1000000 -> shard1）
-- 2. 哈希分片：hash(user_id) % N
-- 3. 一致性哈希：减少数据迁移

-- 分片键选择原则
-- 1. 高基数（cardinality）字段
-- 2. 查询频繁的字段
-- 3. 数据分布均匀

-- 示例：用户表分片
CREATE TABLE users_0 (
    id BIGINT PRIMARY KEY,
    user_id BIGINT,
    name VARCHAR(100)
) PARTITION BY HASH(user_id) PARTITIONS 16;

-- 或使用中间件（ShardingSphere、Vitess）
```

**3. 分布式事务解决方案**

**XA协议（两阶段提交）**：
```sql
-- MySQL支持XA事务
XA START 'xid1';
INSERT INTO orders VALUES (1, 'order1');
XA END 'xid1';
XA PREPARE 'xid1';
-- 其他数据库执行相同操作
XA COMMIT 'xid1';  -- 或 XA ROLLBACK 'xid1'

-- 问题：性能差，存在协调者单点故障
```

**TCC（Try-Confirm-Cancel）模式**：
```sql
-- 业务层实现，非数据库原生支持
-- Try阶段：资源预留
-- Confirm阶段：确认提交
-- Cancel阶段：回滚释放

-- 示例：转账业务
-- Try：冻结账户余额
UPDATE accounts SET frozen = frozen + amount WHERE user_id = ?;
-- Confirm：实际扣款
UPDATE accounts SET balance = balance - amount WHERE user_id = ?;
-- Cancel：解冻
UPDATE accounts SET frozen = frozen - amount WHERE user_id = ?;
```

**Saga模式**：
```sql
-- 长事务分解为多个本地事务
-- 每个事务都有对应的补偿操作

-- 示例：订单流程
-- 1. 创建订单（本地事务）
INSERT INTO orders VALUES (...);
-- 2. 扣减库存（本地事务）
UPDATE inventory SET stock = stock - 1 WHERE product_id = ?;
-- 3. 如果失败，执行补偿操作
-- 补偿1：删除订单
DELETE FROM orders WHERE id = ?;
-- 补偿2：恢复库存
UPDATE inventory SET stock = stock + 1 WHERE product_id = ?;
```

**Seata分布式事务框架**：
```java
// AT模式（自动补偿）
@GlobalTransactional
public void createOrder() {
    // 业务逻辑
    orderService.create();
    inventoryService.deduct();
    // 自动记录undo log，失败时自动回滚
}

// TCC模式
@TwoPhaseBusinessAction(name = "prepare", commitMethod = "commit", rollbackMethod = "rollback")
public boolean prepare(BusinessActionContext context) {
    // Try阶段
    return true;
}
```

#### 3.2 读写分离与高可用

**1. 主从复制（Replication）**
```sql
-- MySQL原生主从复制
-- 异步复制（默认）
-- 半同步复制（MySQL 5.7+）
-- 组复制（Group Replication，MySQL 5.7+）

-- 配置主从
-- 主库配置
server-id = 1
log_bin = mysql-bin
binlog_format = ROW  -- 推荐ROW格式

-- 从库配置
server-id = 2
relay_log = mysql-relay-bin
read_only = 1

-- 查看复制状态
SHOW SLAVE STATUS\G
-- 关键字段：
-- Slave_IO_Running: Yes
-- Slave_SQL_Running: Yes
-- Seconds_Behind_Master: 0
```

**2. 高可用方案**

**MHA（Master High Availability）**：
```
架构：
    MHA Manager（监控节点）
    ├── Master（主库）
    ├── Slave1（从库1）
    └── Slave2（从库2）

故障切换流程：
1. 检测主库故障
2. 选择新主库（优先选择数据最新的从库）
3. 应用差异日志
4. 切换VIP到新主库
5. 重新配置其他从库
```

**Orchestrator**：
```bash
# 自动故障检测和恢复
orchestrator -c discover -i mysql-host:3306
orchestrator -c relocate -i slave-host:3306 --destination master-host:3306

# 支持多种拓扑结构
# - 主从复制
# - 双主复制
# - 级联复制
```

**MySQL Group Replication（MGR）**：
```sql
-- MySQL 5.7+ 内置的高可用方案
-- 基于Paxos协议，强一致性

-- 配置组复制
SET GLOBAL group_replication_bootstrap_group=ON;
START GROUP_REPLICATION;
SET GLOBAL group_replication_bootstrap_group=OFF;

-- 查看组状态
SELECT * FROM performance_schema.replication_group_members;

-- 特点：
-- 1. 自动故障检测和选举
-- 2. 多主模式（可选）
-- 3. 数据强一致性
-- 4. 自动成员管理
```

**3. 读写分离中间件**

**ProxySQL**：
```sql
-- 配置读写分离规则
INSERT INTO mysql_query_rules (
    rule_id,
    active,
    match_digest,
    destination_hostgroup,
    apply
) VALUES (
    1,
    1,
    '^SELECT.*FOR UPDATE',
    10,  -- 写组
    1
);

INSERT INTO mysql_query_rules (
    rule_id,
    active,
    match_digest,
    destination_hostgroup,
    apply
) VALUES (
    2,
    1,
    '^SELECT',
    20,  -- 读组
    1
);

-- 健康检查
UPDATE mysql_servers SET max_replication_lag = 10 WHERE hostgroup_id = 20;
```

**ShardingSphere**：
```yaml
# 配置分片规则
shardingRule:
  tables:
    orders:
      actualDataNodes: ds_${0..1}.orders_${0..15}
      tableStrategy:
        standard:
          shardingColumn: order_id
          preciseAlgorithmClassName: com.example.OrderShardingAlgorithm
  bindingTables:
    - orders,order_items
```

### 四、性能优化与调优

#### 4.1 索引优化策略

**1. 索引设计原则**
```sql
-- 1. 高选择性列优先
-- 选择性 = 不同值数量 / 总行数
-- 选择性 > 0.1 适合建索引

-- 2. 联合索引遵循最左前缀
CREATE INDEX idx_a_b_c ON table(a, b, c);
-- 有效：WHERE a=1, WHERE a=1 AND b=2, WHERE a=1 AND b=2 AND c=3
-- 无效：WHERE b=2, WHERE c=3, WHERE b=2 AND c=3

-- 3. 覆盖索引减少回表
CREATE INDEX idx_cover ON orders(user_id, order_date, amount);
SELECT user_id, order_date, amount FROM orders
WHERE user_id = 123 AND order_date > '2024-01-01';
-- 索引包含所有查询字段，无需回表

-- 4. 避免冗余索引
-- 删除重复索引：idx_a, idx_a_b (idx_a冗余)
```

**2. 索引失效场景**
```sql
-- 1. 隐式类型转换
WHERE user_id = '123'  -- user_id是INT类型，索引失效
-- 应改为：WHERE user_id = 123

-- 2. 函数操作
WHERE YEAR(create_time) = 2024  -- 索引失效
-- 应改为：WHERE create_time >= '2024-01-01' AND create_time < '2025-01-01'

-- 3. 模糊查询前缀
WHERE name LIKE '%Alice'  -- 索引失效
WHERE name LIKE 'Alice%'  -- 索引有效

-- 4. OR条件（部分列无索引）
WHERE a = 1 OR b = 2  -- 如果b无索引，可能全表扫描
-- 应改为：UNION ALL 或 分别查询

-- 5. 范围查询后列索引失效
WHERE a > 10 AND b = 20  -- a是范围，b索引可能失效
```

#### 4.2 查询优化

**1. 执行计划分析**
```sql
EXPLAIN SELECT * FROM orders WHERE user_id = 123;

-- 关键字段解读
-- type: ALL（全表扫描）< index（索引扫描）< range（范围扫描）< ref（索引查找）< const（常量）
-- key: 实际使用的索引
-- rows: 预估扫描行数
-- Extra: 额外信息（Using index, Using where, Using filesort）

-- 优化示例
-- 问题查询
EXPLAIN SELECT * FROM orders
WHERE DATE(create_time) = '2024-01-01';
-- type: ALL, rows: 全表行数

-- 优化后
EXPLAIN SELECT * FROM orders
WHERE create_time >= '2024-01-01' AND create_time < '2024-01-02';
-- type: range, rows: 大幅减少
```

**2. 慢查询优化**
```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = 1;
SET GLOBAL long_query_time = 1;  -- 超过1秒记录
SET GLOBAL slow_query_log_file = '/var/log/mysql/slow.log';

-- 分析慢查询
mysqldumpslow /var/log/mysql/slow.log
pt-query-digest /var/log/mysql/slow.log

-- 常见慢查询模式
-- 1. 全表扫描
-- 2. 临时表（Using temporary）
-- 3. 文件排序（Using filesort）
-- 4. 大结果集（rows过多）
```

**3. 性能模式（Performance Schema）**
```sql
-- MySQL 5.6+ 内置性能监控
-- 查看等待事件
SELECT * FROM performance_schema.events_waits_current;

-- 查看SQL执行统计
SELECT
    DIGEST_TEXT,
    COUNT_STAR,
    AVG_TIMER_WAIT/1000000000 as avg_time_ms,
    SUM_ROWS_EXAMINED
FROM performance_schema.events_statements_summary_by_digest
ORDER BY AVG_TIMER_WAIT DESC
LIMIT 10;

-- 查看锁等待
SELECT
    t.PROCESSLIST_ID,
    t.PROCESSLIST_USER,
    t.PROCESSLIST_HOST,
    t.PROCESSLIST_DB,
    t.PROCESSLIST_COMMAND,
    t.PROCESSLIST_TIME,
    t.PROCESSLIST_INFO
FROM performance_schema.threads t
WHERE t.PROCESSLIST_ID IN (
    SELECT blocking_process_id
    FROM performance_schema.data_lock_waits
);
```

**4. sys schema（MySQL 5.7+）**
```sql
-- 系统视图，简化性能分析
-- 查看慢查询
SELECT * FROM sys.statements_with_temp_tables;
SELECT * FROM sys.statements_with_sorting;

-- 查看索引使用情况
SELECT * FROM sys.schema_index_statistics;

-- 查看表大小和行数
SELECT * FROM sys.schema_table_statistics;

-- 查看未使用的索引
SELECT * FROM sys.schema_unused_indexes;
```

#### 4.3 配置调优

**1. Buffer Pool调优**
```ini
# my.cnf 配置
[mysqld]
# Buffer Pool大小（通常为物理内存的50-70%）
innodb_buffer_pool_size = 16G

# Buffer Pool实例数（多实例减少竞争）
innodb_buffer_pool_instances = 8

# 页大小（默认16K，适合OLTP）
innodb_page_size = 16K

# 预读策略
innodb_read_ahead_threshold = 56  # 连续访问56页后触发预读
innodb_random_read_ahead = OFF    # 关闭随机预读

# 刷新策略
innodb_flush_log_at_trx_commit = 1  # 1: 每次提交都刷盘（最安全）
innodb_flush_method = O_DIRECT      # 绕过OS缓存，直接写磁盘
```

**2. 事务和日志调优**
```ini
# Redo Log
innodb_log_file_size = 2G          # 每个日志文件大小
innodb_log_buffer_size = 64M       # 日志缓冲区大小
innodb_log_files_in_group = 3      # 日志文件组数量

# 事务隔离级别
transaction_isolation = REPEATABLE-READ  # 默认，适合大多数场景

# 死锁检测
innodb_deadlock_detect = ON        # 开启死锁检测
innodb_lock_wait_timeout = 50      # 锁等待超时时间（秒）

# 自适应哈希索引
innodb_adaptive_hash_index = ON    # 开启AHI（适合点查询多的场景）
```

**3. 连接和线程调优**
```ini
# 连接配置
max_connections = 1000             # 最大连接数
max_connect_errors = 100           # 最大连接错误数
connect_timeout = 10               # 连接超时（秒）

# 线程缓存
thread_cache_size = 50             # 线程缓存数量
thread_handling = pool-of-threads  # 线程池模式（MySQL 5.7+）

# 临时表配置
tmp_table_size = 64M               # 内存临时表大小
max_heap_table_size = 64M          # 内存表最大大小
```

### 五、与Redis/Elasticsearch协同

#### 5.1 MySQL + Redis缓存模式

**1. 缓存策略**
```sql
-- 1. Cache-Aside模式（最常用）
-- 应用层控制缓存
def get_user(user_id):
    # 1. 查缓存
    user = redis.get(f"user:{user_id}")
    if user:
        return json.loads(user)

    # 2. 查数据库
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)

    # 3. 写缓存
    redis.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user

-- 2. Read-Through模式
-- 缓存中间件自动处理
-- 3. Write-Through模式
-- 写操作同时更新缓存和数据库
```

**2. 缓存一致性**
```sql
-- 问题：数据库更新后，缓存未更新
-- 解决方案：
-- 1. 删除缓存（Cache-Aside常用）
UPDATE users SET name = 'Alice' WHERE id = 1;
DEL user:1;  -- 删除缓存

-- 2. 延迟双删（解决并发问题）
UPDATE users SET name = 'Alice' WHERE id = 1;
DEL user:1;
sleep(500);  -- 延迟500ms
DEL user:1;

-- 3. Canal监听binlog
-- Canal解析binlog，自动更新缓存
```

**3. 热点数据处理**
```sql
-- 1. 缓存预热
-- 系统启动时加载热点数据到缓存
SELECT * FROM hot_products WHERE status = 'active';

-- 2. 缓存分层
-- L1: 本地缓存（Caffeine/Guava）
-- L2: Redis集群
-- L3: MySQL

-- 3. 缓存穿透保护
-- 布隆过滤器防止查询不存在的数据
-- 空值缓存（短时间）
```

#### 5.2 MySQL + Elasticsearch

**1. 数据同步方案**
```sql
-- 1. Logstash同步
-- 配置文件：logstash.conf
input {
  jdbc {
    jdbc_driver_library => "mysql-connector-java.jar"
    jdbc_driver_class => "com.mysql.jdbc.Driver"
    jdbc_connection_string => "jdbc:mysql://localhost:3306/mydb"
    statement => "SELECT * FROM products WHERE updated_at > :sql_last_value"
    schedule => "* * * * *"  # 每分钟同步
  }
}
output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "products"
    document_id => "%{id}"
  }
}

-- 2. Canal同步（推荐）
-- Canal模拟MySQL slave，监听binlog
-- 支持全量同步和增量同步

-- 3. 应用层双写
-- 写MySQL同时写ES
def create_product(product):
    # 写MySQL
    product_id = db.insert("products", product)

    # 写ES
    es.index(index="products", id=product_id, body=product)

    return product_id
```

**2. 搜索优化**
```sql
-- MySQL全文索引 vs Elasticsearch
-- MySQL全文索引（MySQL 5.6+）
CREATE FULLTEXT INDEX idx_ft ON articles(title, content);
SELECT * FROM articles
WHERE MATCH(title, content) AGAINST('database' IN NATURAL LANGUAGE MODE);

-- Elasticsearch优势：
-- 1. 更复杂的查询（模糊、聚合、高亮）
-- 2. 更好的分词器（IK、Jieba）
-- 3. 分布式搜索
-- 4. 实时索引

-- 混合架构：MySQL主存储 + ES搜索
-- MySQL：事务性操作、强一致性
-- ES：搜索、分析、聚合
```

### 六、学习笔记与踩坑记录

#### 6.1 重要理解

**1. InnoDB的ACID实现**
- **原子性**：通过Undo Log实现
- **一致性**：通过Redo Log + Undo Log保证
- **隔离性**：通过MVCC + 锁机制实现
- **持久性**：通过Redo Log + Doublewrite Buffer实现

**2. MVCC的局限性**
- 只在REPEATABLE READ和READ COMMITTED级别有效
- 长事务会导致Undo Log膨胀
- 大量历史版本影响性能

**3. 索引不是越多越好**
- 每个索引增加写开销（插入、更新、删除）
- 索引占用存储空间
- 优化器可能选择错误的索引

**4. 分布式事务的权衡**
- XA：强一致性，性能差
- TCC：业务侵入性强，但性能好
- Saga：适合长事务，需要补偿机制

#### 6.2 踩坑记录

**1. 大表DDL操作**
```sql
-- 问题：直接ALTER TABLE大表会导致锁表
-- 解决方案：
-- 1. 使用pt-online-schema-change
pt-online-schema-change --alter "ADD COLUMN new_col VARCHAR(100)" D=mydb,t=large_table --execute

-- 2. 使用gh-ost（GitHub开源）
gh-ost --user="root" --database="mydb" --table="large_table" \
  --alter="ADD COLUMN new_col VARCHAR(100)" --execute

-- 3. 分批处理
-- 对于数据迁移，分批迁移
```

**2. 深分页问题**
```sql
-- 问题：OFFSET过大时性能差
SELECT * FROM orders ORDER BY id LIMIT 1000000, 10;

-- 解决方案：
-- 1. 使用游标（推荐）
SELECT * FROM orders
WHERE id > 1000000
ORDER BY id
LIMIT 10;

-- 2. 延迟关联
SELECT * FROM orders o
JOIN (SELECT id FROM orders ORDER BY id LIMIT 1000000, 10) t
ON o.id = t.id;

-- 3. 业务层限制分页深度
```

**3. 大事务问题**
```sql
-- 问题：单个事务操作过多数据，导致锁等待、Undo Log膨胀
-- 解决方案：
-- 1. 分批提交
START TRANSACTION;
UPDATE large_table SET status = 'processed' WHERE id BETWEEN 1 AND 1000;
COMMIT;
START TRANSACTION;
UPDATE large_table SET status = 'processed' WHERE id BETWEEN 1001 AND 2000;
COMMIT;

-- 2. 使用存储过程分批
DELIMITER $$
CREATE PROCEDURE batch_update()
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE batch_size INT DEFAULT 1000;
    DECLARE start_id INT DEFAULT 1;
    DECLARE max_id INT;

    SELECT MAX(id) INTO max_id FROM large_table;

    WHILE start_id <= max_id DO
        UPDATE large_table
        SET status = 'processed'
        WHERE id BETWEEN start_id AND start_id + batch_size - 1;

        SET start_id = start_id + batch_size;
        COMMIT;
    END WHILE;
END$$
DELIMITER ;
```

**4. 索引失效的隐蔽场景**
```sql
-- 1. OR条件中部分列无索引
WHERE indexed_col = 1 OR unindexed_col = 2
-- 解决方案：UNION ALL 或 分别查询

-- 2. 负向查询
WHERE id NOT IN (1, 2, 3)  -- 索引失效
WHERE id != 1              -- 索引失效
-- 解决方案：改写为范围查询

-- 3. 函数作用于索引列
WHERE DATE(create_time) = '2024-01-01'
-- 解决方案：改写为范围查询

-- 4. 隐式类型转换
WHERE phone = '13800138000'  -- phone是VARCHAR，但传入数字
-- 解决方案：确保类型一致
```

### 七、后续行动计划

#### 7.1 短期计划（1-2周）

**1. 深入学习MySQL 8.0新特性**
- [ ] 实践窗口函数在复杂报表中的应用
- [ ] 使用CTE重构复杂查询
- [ ] 测试JSON数据类型的性能和使用场景
- [ ] 配置直方图统计信息并观察查询优化效果

**2. 性能调优实战**
- [ ] 在mylearn目录创建测试环境
- [ ] 设计压力测试场景（模拟高并发）
- [ ] 使用Performance Schema和sys schema分析性能瓶颈
- [ ] 实践索引优化和查询重写

**3. 分布式方案验证**
- [ ] 搭建MySQL主从复制环境
- [ ] 测试MHA或Orchestrator高可用方案
- [ ] 实现简单的分库分表（使用ShardingSphere）
- [ ] 测试分布式事务（Seata）

#### 7.2 中期计划（3-4周）

**1. 实战项目集成**
- [ ] 在高性能API网关项目中集成MySQL
- [ ] 实现读写分离和缓存策略
- [ ] 设计分库分表方案（如果数据量大）
- [ ] 配置监控和告警（Prometheus + Grafana）

**2. 与AI/ML系统结合**
- [ ] 设计向量数据库与MySQL的协同方案
- [ ] 实现AI模型训练数据的MySQL存储和查询优化
- [ ] 探索MySQL在Agent基础设施中的应用

**3. 性能基准测试**
- [ ] 对比不同索引策略的性能差异
- [ ] 测试不同事务隔离级别的性能影响
- [ ] 评估MySQL 8.0 vs 5.7的性能提升

#### 7.3 长期目标

**1. 成为MySQL专家**
- [ ] 深入理解InnoDB源码（可选）
- [ ] 掌握MySQL内核调优
- [ ] 参与MySQL社区贡献

**2. 架构设计能力**
- [ ] 设计支持千万级日活的数据库架构
- [ ] 实现高可用、高并发的MySQL集群
- [ ] 优化大规模数据处理Pipeline的数据库支撑

**3. 技术影响力**
- [ ] 撰写技术博客分享MySQL经验
- [ ] 在团队内部进行技术分享
- [ ] 参与技术社区交流

## 总结

MySQL作为关系型数据库的代表，在分布式系统中仍然扮演着核心角色。对于资深分布式工程师，需要：

1. **深入理解经典机制**：MVCC、锁、索引、事务等核心原理
2. **掌握现代特性**：MySQL 8.0+的窗口函数、CTE、JSON增强等
3. **具备分布式能力**：分库分表、读写分离、高可用、分布式事务
4. **精通性能优化**：索引设计、查询优化、配置调优
5. **善于系统集成**：与Redis、Elasticsearch等协同工作

本笔记提供了从经典到现代的MySQL知识体系，结合`fullstack-jd.md`中的职责要求，为处理高并发、大规模数据处理、分布式系统等场景打下坚实基础。

---

**参考资源**：
- MySQL官方文档：https://dev.mysql.com/doc/
- MySQL 8.0新特性：https://dev.mysql.com/doc/refman/8.0/en/
- InnoDB存储引擎：https://dev.mysql.com/doc/refman/8.0/en/innodb-storage-engine.html
- 性能优化指南：https://dev.mysql.com/doc/refman/8.0/en/optimization.html
