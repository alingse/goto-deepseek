# Redis Command 参考手册

**日期**: 2026-02-12
**学习路径**: 10 - 分布式存储与消息系统
**对话主题**: Redis 所有核心命令详解

---

## 一、String 命令

### 基础操作

| 命令 | 语法 | 时间复杂度 | 说明 |
|-----|------|----------|------|
| **SET** | `SET key value [NX\|XX] [EX seconds\|PX milliseconds]` | O(1) | 设置值，NX=不存在才设，XX=存在才设 |
| **GET** | `GET key` | O(1) | 获取值 |
| **MSET** | `MSET key value [key value ...]` | O(n) | 批量设置 |
| **MGET** | `MGET key [key ...]` | O(n) | 批量获取 |
| **DEL** | `DEL key [key ...]` | O(n) | 删除键 |
| **EXISTS** | `EXISTS key [key ...]` | O(n) | 键是否存在 |
| **TYPE** | `TYPE key` | O(1) | 获取值类型 |
| **EXPIRE** | `EXPIRE key seconds` | O(1) | 设置过期时间 |
| **TTL** | `TTL key` | O(1) | 剩余生存时间 |
| **PERSIST** | `PERSIST key` | O(1) | 移除过期时间 |

### 数值操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **INCR** | `INCR key` | 原子 +1 |
| **DECR** | `DECR key` | 原子 -1 |
| **INCRBY** | `INCRBY key increment` | 原子 +n |
| **DECRBY** | `DECRBY key decrement` | 原子 -n |
| **INCRBYFLOAT** | `INCRBYFLOAT key increment` | 原子 +浮点数 |

### 字符串操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **APPEND** | `APPEND key value` | 追加字符串 |
| **STRLEN** | `STRLEN key` | 获取长度 |
| **GETRANGE** | `GETRANGE key start end` | 获取子串 (O(n)) |
| **SETRANGE** | `SETRANGE key offset value` | 覆盖子串 |
| **GETSET** | `GETSET key value` | 原子设置并返回旧值 |

### 位操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **SETBIT** | `SETBIT key offset value` | 设置位 |
| **GETBIT** | `GETBIT key offset` | 获取位 |
| **BITCOUNT** | `BITCOUNT key [start end]` | 统计1位数 |
| **BITPOS** | `BITPOS key bit [start end]` | 查找第一个0/1位 |
| **BITOP** | `BITOP operation destkey key [key ...]` | 位运算 AND/OR/XOR/NOT |

---

## 二、Hash 命令

### 基础操作

| 命令 | 语法 | 时间复杂度 | 说明 |
|-----|------|----------|------|
| **HSET** | `HSET key field value [field value ...]` | O(n) | 设置哈希字段 |
| **HGET** | `HGET key field` | O(1) | 获取字段值 |
| **HMGET** | `HMGET key field [field ...]` | O(n) | 批量获取 |
| **HGETALL** | `HGETALL key` | O(n) | 获取所有字段和值 |
| **HDEL** | `HDEL key field [field ...]` | O(n) | 删除字段 |
| **HEXISTS** | `HEXISTS key field` | O(1) | 字段是否存在 |
| **HLEN** | `HLEN key` | O(1) | 字段数量 |

### 高级操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **HINCRBY** | `HINCRBY key field increment` | 字段值 +n |
| **HINCRBYFLOAT** | `HINCRBYFLOAT key field increment` | 字段值 +浮点 |
| **HKEYS** | `HKEYS key` | 获取所有字段名 |
| **HVALS** | `HVALS key` | 获取所有值 |
| **HSCAN** | `HSCAN key cursor [MATCH pattern] [COUNT count]` | 渐进遍历 |
| **HSTRLEN** | `HSTRLEN key field` | 获取字段值长度 |

---

## 三、List 命令

### 基础操作

| 命令 | 语法 | 时间复杂度 | 说明 |
|-----|------|----------|------|
| **LPUSH** | `LPUSH key value [value ...]` | O(1) | 头部插入 |
| **RPUSH** | `RPUSH key value [value ...]` | O(1) | 尾部插入 |
| **LPOP** | `LPOP key [count]` | O(1) | 头部弹出 |
| **RPOP** | `RPOP key [count]` | O(1) | 尾部弹出 |
| **LRANGE** | `LRANGE key start stop` | O(n) | 范围获取 |
| **LLEN** | `LLEN key` | O(1) | 列表长度 |

### 阻塞操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **BLPOP** | `BLPOP key [key ...] timeout` | 阻塞左弹出 |
| **BRPOP** | `BRPOP key [key ...] timeout` | 阻塞右弹出 |
| **BRPOPLPUSH** | `BRPOPLPUSH source destination timeout` | 阻塞右弹左推 |
| **BLMOVE** | `BLMOVE source destination LEFT\|RIGHT LEFT\|RIGHT timeout` | 阻塞移动 |

### 高级操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **LINDEX** | `LINDEX key index` | 按索引获取 |
| **LSET** | `LSET key index value` | 按索引设置 |
| **LINSERT** | `LINSERT key BEFORE\|AFTER pivot value` | 插入元素 |
| **LREM** | `LREM key count value` | 移除元素 |
| **LTRIM** | `LTRIM key start stop` | 修剪列表 |
| **RPOPLPUSH** | `RPOPLPUSH source destination` | 右弹左推 |

---

## 四、Set 命令

### 基础操作

| 命令 | 语法 | 时间复杂度 | 说明 |
|-----|------|----------|------|
| **SADD** | `SADD key member [member ...]` | O(n) | 添加成员 |
| **SREM** | `SREM key member [member ...]` | O(n) | 移除成员 |
| **SMEMBERS** | `SMEMBERS key` | O(n) | 获取所有成员 |
| **SISMEMBER** | `SISMEMBER key member` | O(1) | 是否存在 |
| **SCARD** | `SCARD key` | O(1) | 成员数量 |
| **SPOP** | `SPOP key [count]` | O(n) | 随机弹出 |

### 集合运算

| 命令 | 语法 | 说明 |
|-----|------|------|
| **SUNION** | `SUNION key [key ...]` | 并集 |
| **SINTER** | `SINTER key [key ...]` | 交集 |
| **SDIFF** | `SDIFF key [key ...]` | 差集 |
| **SUNIONSTORE** | `SUNIONSTORE dest key [key ...]` | 并集存储 |
| **SINTERSTORE** | `SINTERSTORE dest key [key ...]` | 交集存储 |
| **SDIFFSTORE** | `SDIFFSTORE dest key [key ...]` | 差集存储 |

### 随机操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **SRANDMEMBER** | `SRANDMEMBER key [count]` | 随机获取 |
| **SSCAN** | `SSCAN key cursor [MATCH pattern] [COUNT count]` | 渐进遍历 |

---

## 五、Sorted Set 命令

### 基础操作

| 命令 | 语法 | 时间复杂度 | 说明 |
|-----|------|----------|------|
| **ZADD** | `ZADD key [NX\|XX] [GT\|LT] [CH] [INCR] score member [score member ...]` | O(log n) | 添加成员 |
| **ZREM** | `ZREM key member [member ...]` | O(log n) | 移除成员 |
| **ZSCORE** | `ZSCORE key member` | O(1) | 获取分数 |
| **ZRANGE** | `ZRANGE key start stop [WITHSCORES]` | O(log n + m) | 按索引范围 |
| **ZREVRANGE** | `ZREVRANGE key start stop [WITHSCORES]` | O(log n + m) | 逆序范围 |
| **ZRANGEBYSCORE** | `ZRANGEBYSCORE key min max [WITHSCORES] [LIMIT offset count]` | O(log n + m) | 按分数范围 |
| **ZREVRANGEBYSCORE** | `ZREVRANGEBYSCORE key max min [WITHSCORES] [LIMIT offset count]` | O(log n + m) | 逆序分数 |

### 排名操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **ZRANK** | `ZRANK key member` | 正序排名 (0-based) |
| **ZREVRANK** | `ZREVRANK key member` | 逆序排名 |
| **ZCOUNT** | `ZCOUNT key min max` | 分数范围数量 |
| **ZCARD** | `ZCARD key` | 成员数量 |

### 集合运算

| 命令 | 语法 | 说明 |
|-----|------|------|
| **ZUNIONSTORE** | `ZUNIONSTORE dest numkeys key [key ...]` | 并集 |
| **ZINTERSTORE** | `ZINTERSTORE dest numkeys key [key ...]` | 交集 |
| **ZDIFFSTORE** | `ZDIFFSTORE dest numkeys key [key ...]` | 差集 |

### 阻塞操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **BZPOPMAX** | `BZPOPMAX key [key ...] timeout` | 阻塞弹最大 |
| **BZPOPMIN** | `BZPOPMIN key [key ...] timeout` | 阻塞弹最小 |

---

## 六、Stream 命令

### 基础操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **XADD** | `XADD key [NOMKSTREAM] [MAXLEN [~\|=] count] [MINID [~\|=] id] *\|id field value [field value ...]` | 添加消息 |
| **XRANGE** | `XRANGE key start end [COUNT count]` | 范围查询 |
| **XREVRANGE** | `XREVRANGE key end start [COUNT count]` | 逆序范围 |
| **XLEN** | `XLEN key` | 消息数量 |
| **XACK** | `XACK key group id [id ...]` | 确认处理 |
| **XDEL** | `XDEL key id [id ...]` | 删除消息 |
| **XTRIM** | `XTRIM key [MINID [~\|=] id] [MAXLEN [~\|=] count]` | 裁剪流 |

### 消费操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **XREAD** | `XREAD [COUNT count] [BLOCK milliseconds] STREAMS key id [key id ...]` | 消费消息 |
| **XREADGROUP** | `XREADGROUP GROUP group consumer [COUNT count] [BLOCK milliseconds] [NOACK] STREAMS key [key ...] id [id ...]` | 组消费 |
| **XPENDING** | `XPENDING key group [start end count] [consumer]` | 待确认消息 |

### 消费者组

| 命令 | 语法 | 说明 |
|-----|------|------|
| **XGROUP CREATE** | `XGROUP CREATE key group id [MKSTREAM] [ENTRIESREAD entries]` | 创建组 |
| **XGROUP DESTROY** | `XGROUP DESTROY key group` | 删除组 |
| **XGROUP CREATECONSUMER** | `XGROUP CREATECONSUMER key group consumer` | 创建消费者 |
| **XGROUP SETID** | `XGROUP SETID key group id [ENTRIESREAD entries]` | 设置起始ID |

---

## 七、Key 命令

| 命令 | 语法 | 说明 |
|-----|------|------|
| **KEYS** | `KEYS pattern` | 模式匹配 (生产禁用) |
| **SCAN** | `SCAN cursor [MATCH pattern] [COUNT count] [TYPE type]` | 渐进遍历 |
| **RENAME** | `RENAME key newkey` | 重命名 |
| **RENAMENX** | `RENAMENX key newkey` | 不存在才重命名 |
| **RANDOMKEY** | `RANDOMKEY` | 随机键 |
| **COPY** | `COPY source destination [DB destination-db] [REPLACE]` | 复制 |
| **DUMP** | `DUMP key` | 序列化 |
| **RESTORE** | `RESTORE key ttl serialized-value [REPLACE]` | 反序列化 |
| **OBJECT ENCODING** | `OBJECT ENCODING key` | 编码类型 |
| **OBJECT IDLETIME** | `OBJECT IDLETIME key` | 空闲时间 |
| **OBJECT FREQ** | `OBJECT FREQ key` | 频率(LRU) |

---

## 八、Transaction 命令

| 命令 | 语法 | 说明 |
|-----|------|------|
| **MULTI** | `MULTI` | 开启事务 |
| **EXEC** | `EXEC` | 执行事务 |
| **DISCARD** | `DISCARD` | 取消事务 |
| **WATCH** | `WATCH key [key ...]` | 乐观锁 |
| **UNWATCH** | `UNWATCH` | 取消监视 |

---

## 九、Pub/Sub 命令

| 命令 | 语法 | 说明 |
|-----|------|------|
| **PUBLISH** | `PUBLISH channel message` | 发布 |
| **SUBSCRIBE** | `SUBSCRIBE channel [channel ...]` | 订阅 |
| **PSUBSCRIBE** | `PSUBSCRIBE pattern [pattern ...]` | 模式订阅 |
| **UNSUBSCRIBE** | `UNSUBSCRIBE [channel [channel ...]]` | 取消订阅 |
| **PUBSUB CHANNELS** | `PUBSUB CHANNELS [pattern]` | 活跃频道 |
| **PUBSUB NUMSUB** | `PUBSUB NUMSUB [channel ...]` | 订阅数 |

---

## 十、Scripting 命令

| 命令 | 语法 | 说明 |
|-----|------|------|
| **EVAL** | `EVAL script numkeys key [key ...] arg [arg ...]` | 执行 Lua |
| **EVALSHA** | `EVALSHA sha1 numkeys key [key ...] arg [arg ...]` | 执行缓存脚本 |
| **SCRIPT LOAD** | `SCRIPT LOAD script` | 加载脚本 |
| **SCRIPT EXISTS** | `SCRIPT EXISTS sha1 [sha1 ...]` | 检查脚本 |
| **SCRIPT FLUSH** | `SCRIPT FLUSH` | 清空脚本缓存 |
| **SCRIPT KILL** | `SCRIPT KILL` | 终止运行中脚本 |
| **FCALL** | `FCALL function_name number_of_keys [key1 key2 ... arg1 arg2 ...]` | 调用函数 |

---

## 十一、Server 命令

### 信息与统计

| 命令 | 语法 | 说明 |
|-----|------|------|
| **INFO** | `INFO [section]` | 服务器信息 |
| **CLIENT LIST** | `CLIENT LIST [TYPE normal\|master\|replica\|pubsub]` | 客户端列表 |
| **CLIENT INFO** | `CLIENT INFO` | 当前客户端信息 |
| **CLIENT KILL** | `CLIENT KILL [ip:port] [ID client-id] [TYPE normal\|master\|replica\|pubsub]` | 关闭客户端 |
| **CLIENT GETNAME** | `CLIENT GETNAME` | 获取客户端名 |
| **CLIENT SETNAME** | `CLIENT SETNAME name` | 设置客户端名 |
| **CONFIG GET** | `CONFIG GET parameter` | 获取配置 |
| **CONFIG SET** | `CONFIG SET parameter value` | 设置配置 |
| **CONFIG RESETSTAT** | `CONFIG RESETSTAT` | 重置统计 |

### 持久化操作

| 命令 | 语法 | 说明 |
|-----|------|------|
| **SAVE** | `SAVE` | 同步保存 RDB |
| **BGSAVE** | `BGSAVE` | 异步保存 RDB |
| **BGREWRITEAOF** | `BGREWRITEAOF` | 异步重写 AOF |
| **LASTSAVE** | `LASTSAVE` | 上次保存时间 |
| **SHUTDOWN** | `SHUTDOWN [NOSAVE\|SAVE]` | 关闭服务器 |

### 慢查询

| 命令 | 语法 | 说明 |
|-----|------|------|
| **SLOWLOG GET** | `SLOWLOG GET [count]` | 获取慢查询 |
| **SLOWLOG LEN** | `SLOWLOG LEN` | 慢查询数量 |
| **SLOWLOG RESET** | `SLOWLOG RESET` | 清空慢查询 |

---

## 十二、Cluster 命令

| 命令 | 语法 | 说明 |
|-----|------|------|
| **CLUSTER INFO** | `CLUSTER INFO` | 集群信息 |
| **CLUSTER NODES** | `CLUSTER NODES` | 节点列表 |
| **CLUSTER SLOTS** | `CLUSTER SLOTS` | 槽分配 |
| **CLUSTER MEET** | `CLUSTER MEET ip port` | 节点加入 |
| **CLUSTER FORGET** | `CLUSTER FORGET node-id` | 移除节点 |
| **CLUSTER REPLICATE** | `CLUSTER REPLICATE node-id` | 设置从节点 |
| **CLUSTER ADDSLOTS** | `CLUSTER ADDSLOTS slot [slot ...]` | 分配槽 |
| **CLUSTER DELSLOTS** | `CLUSTER DELSLOTS slot [slot ...]` | 删除槽 |
| **CLUSTER SETSLOT** | `CLUSTER SETSLOT slot IMPORTING\|MIGRATING\|NODE node-id\|STABLE` | 槽迁移 |
| **CLUSTER SLAVES** | `CLUSTER SLAVES node-id` | 获取从节点 |
| **ASKING** | `ASKING` | 询问转向 |

---

## 十三、ACL 命令

| 命令 | 语法 | 说明 |
|-----|------|------|
| **ACL LIST** | `ACL LIST` | 列出所有规则 |
| **ACL USER** | `ACL USER username` | 查看用户 |
| **ACL SETUSER** | `ACL SETUSER username [rule [rule ...]]` | 设置用户 |
| **ACL GETUSER** | `ACL GETUSER username` | 获取用户规则 |
| **ACL DELUSER** | `ACL DELUSER username [username ...]` | 删除用户 |
| **ACL WHOAMI** | `ACL WHOAMI` | 当前用户 |
| **ACL LOG** | `ACL LOG [count]` | 安全日志 |

---

## 十四、连接命令

| 命令 | 语法 | 说明 |
|-----|------|------|
| **AUTH** | `AUTH [username] password` | 认证 |
| **SELECT** | `SELECT index` | 切换数据库 |
| **ECHO** | `ECHO message` | 回显 |
| **PING** | `PING [message]` | 心跳 |
| **QUIT** | `QUIT` | 退出 |
| **FLUSHDB** | `FLUSHDB [ASYNC]` | 清空当前库 |
| **FLUSHALL** | `FLUSHALL [ASYNC]` | 清空所有库 |
| **DBSIZE** | `DBSIZE` | 当前库键数量 |
| **TIME** | `TIME` | 当前时间 |

---

## 十五、Redis 8.x 新增命令

| 命令 | 说明 |
|-----|------|
| **HOTKEYS** | 查找热键 (Redis 8.6) |
| **KEYS MEMORY HISTOGRAMS** | 键内存分布 (Redis 8.6) |
| **CLUSTER SLOT-STATS** | 槽统计 (Redis 8.6) |

---

## 十六、Redis Stack 扩展命令

### 向量搜索 (Vector Set)

| 命令 | 语法 | 说明 |
|-----|------|------|
| **VSS.CREATE** | `VSS.CREATE key [TYPE HNSW] [DIM dimension]` | 创建向量索引 |
| **VSS.ADD** | `VSS.ADD key score member vector` | 添加向量 |
| **VSS.SEARCH** | `VSS.SEARCH key vector [TOPK k]` | 向量搜索 |
| **VSS.DELETE** | `VSS.DELETE key member` | 删除向量 |

### JSON

| 命令 | 语法 | 说明 |
|-----|------|------|
| **JSON.SET** | `JSON.SET key path json` | 设置 JSON |
| **JSON.GET** | `JSON.GET key [path]` | 获取 JSON |
| **JSON.DEL** | `JSON.DEL key [path]` | 删除 JSON |
| **JSON.NUMINCRBY** | `JSON.NUMINCRBY key path increment` | 数值增加 |
| **JSON.ARRAPPEND** | `JSON.ARRAPPEND key path json` | 数组追加 |

### Time Series

| 命令 | 语法 | 说明 |
|-----|------|------|
| **TS.CREATE** | `TS.CREATE key [RETENTION secs]` | 创建时序 |
| **TS.ADD** | `TS.ADD key timestamp value` | 添加数据 |
| **TS.RANGE** | `TS.RANGE key start end` | 范围查询 |
| **TS.GET** | `TS.GET key` | 获取最后值 |

### Search

| 命令 | 语法 | 说明 |
|-----|------|------|
| **FT.CREATE** | `FT.CREATE index ON HASH SCHEMA field` | 创建索引 |
| **FT.SEARCH** | `FT.SEARCH index query` | 搜索 |
| **FT.AGGREGATE** | `FT.AGGREGATE index query` | 聚合查询 |
| **FT.INFO** | `FT.INFO index` | 索引信息 |

---

## 附录：命令时间复杂度速查

```
O(1):  SET, GET, DEL, EXISTS, TYPE, INCR, DECR, HSET, HGET, HEXISTS,
       LPUSH, RPUSH, LPOP, RPOP, LLEN, SADD, SREM, SISMEMBER, SCARD,
       ZADD, ZREM, ZSCORE, ZRANK, ZCARD

O(log n):  ZADD, ZREM, ZRANGEBYSCORE, ZRANK, ZREVRANK

O(n):  GETRANGE, SETRANGE, HGETALL, HMGET, SMEMBERS, LRANGE, ZRANGE,
       KEYS, DEL, EXPIRE

O(n + m):  Set/Hash/ZSet 的集合运算
```
