# Redis 源代码日期**: 202实现分析

**6-02-12
**学习路径**: 10 - 分布式存储与消息系统
**对话主题**: Redis 核心源码结构与实现机制

---

## 一、Redis 源码目录结构

```
redis/src/
├── server.c             # 主服务器程序入口
├── server.h             # 服务器核心数据结构
├── anet.c               # 网络通信封装
├── ae.c                 # 事件循环 (Redis自己的事件库)
├── networking.c         # 网络协议解析
├── object.c             # Redis 对象实现
│
├── t_string.c           # String 数据类型实现
├── t_hash.c             # Hash 数据类型实现
├── t_list.c             # List 数据类型实现
├── t_set.c              # Set 数据类型实现
├── t_zset.c             # Sorted Set 数据类型实现
├── t_stream.c           # Stream 数据类型实现
│
├── dict.c               # 字典 (Hash表) 实现
├── sds.c                # Simple Dynamic String 实现
├── quicklist.c          # List 底层实现
├── ziplist.c            # 压缩列表实现
├── skiplist.c           # 跳表实现
├── intset.c             # 整数集合实现
│
├── redis-cli.c          # 客户端程序
├── cluster.c            # Redis Cluster 实现
├── replication.c       # 主从复制实现
├── aof.c                # AOF 持久化实现
├── rdb.c                # RDB 持久化实现
├── bio.c                # 后台 I/O 操作
├── latency.c            # 延迟监控
├── lazyfree.c           # 惰性删除
│
└── Makefile             # 编译配置
```

---

## 二、核心数据结构

### 2.1 redisObject（redisObject）

```c
// src/server.h
typedef struct redisObject {
    unsigned type:4;        // 数据类型 (4位)
    unsigned encoding:4;    // 编码方式 (4位)
    unsigned lru:LRU_BITS;  // LRU 信息 (24位)
    int refcount;          // 引用计数
    void *ptr;             // 指向实际数据的指针
} robj;
```

#### type 字段

```c
// src/server.h
#define OBJ_STRING 0
#define OBJ_LIST 1
#define OBJ_SET 2
#define OBJ_ZSET 3
#define OBJ_HASH 4
#define OBJ_STREAM 5
#define OBJ_MODULE 6
```

#### encoding 字段

```c
// src/server.h
// String 编码
#define OBJ_ENCODING_RAW 0     // 原始 SDS
#define OBJ_ENCODING_INT 1     // 整数

// Hash 编码
#define OBJ_ENCODING_ZIPLIST 5 // 压缩列表
#define OBJ_ENCODING_HT 2      // 哈希表

// List 编码
#define OBJ_ENCODING_ZIPLIST 5 // 压缩列表
#define OBJ_ENCODING_QUICKLIST 7 // QuickList

// Set 编码
#define OBJ_ENCODING_INTSET 6  // 整数集合
#define OBJ_ENCODING_HT 2      // 哈希表

// Sorted Set 编码
#define OBJ_ENCODING_ZIPLIST 5 // 压缩列表
#define OBJ_ENCODING_SKIPLIST 8 // 跳表
```

---

### 2.2 SDS (Simple Dynamic String)

```c
// src/sds.h
struct sdshdr {
    int len;        // 已使用长度
    int alloc;     // 分配的空间
    unsigned char flags;  // 标志
    char buf[];    // 柔性数组，存储实际数据
};

// 获取实际字符串指针
static inline char *sdsPtr(const sds s) {
    return (char*)(s - sizeof(struct sdshdr));
}
```

#### SDS 特性

```
┌─────────────────────────────────────────────────────────────┐
│                      SDS 内存布局                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   struct sdshdr:                                           │
│   ┌────────┬────────┬──────┬─────────────────────────┐     │
│   │  len   │  alloc │ flag │        buf[]           │     │
│   │ (4byte)│ (4byte)│(1B)  │   (柔性数组，真实数据)  │     │
│   └────────┴────────┴──────┴─────────────────────────┘     │
│                                                             │
│   优点：                                                    │
│   1. O(1) 获取长度 (len 字段)                             │
│   2. 二进制安全 (不依赖 \0 判断结束)                       │
│   3. 空间预分配 (减少内存分配次数)                         │
│   4. 惰性释放 (不立即回收空间)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.3 Dict (字典/哈希表)

```c
// src/dict.h
typedef struct dict {
    dictType *type;
    dictEntry **table;         // 哈希表数组
    unsigned long size;        // 哈希表大小
    unsigned long used;        // 已使用条目数
    unsigned long rehashidx;   // rehash 进度，-1表示未开始
    int iterators;             // 正在遍历的数量
} dict;

typedef struct dictEntry {
    void *key;
    union {
        void *val;
        uint64_t u64;
        int64_t s64;
        double d;
    } v;
    struct dictEntry *next;    // 链地址法解决冲突
} dictEntry;
```

#### 渐进式 Rehash

```c
// src/dict.c
int dictRehash(dict *d, int n) {
    // 每次只迁移 n 个 bucket
    // 每次 dictAddRaw/dictFind 等操作时触发

    while (n--) {
        dictEntry *de, *nextde;

        // 跳过空 bucket
        if (d->rehashidx >= (signed)d->ht_table[0].size)
            return 1;  // 迁移完成

        // 找到非空 bucket
        de = d->ht_table[0].table[d->rehashidx];
        while(de) {
            // 迁移所有元素到 ht[1]
            // ...
            d->rehashidx++;
        }
    }
    return 0;
}
```

```
┌─────────────────────────────────────────────────────────────┐
│                   渐进式 Rehash 流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   初始状态:                                                 │
│   ┌────────────────────┐  ┌────────────────────┐          │
│   │    ht[0] (旧表)    │  │    ht[1] (新表)    │          │
│   │  [0]→a→b          │  │     空             │          │
│   │  [1]→c            │  │     空             │          │
│   │  [2]              │  │     空             │          │
│   └────────────────────┘  └────────────────────┘          │
│   rehashidx = 0                                            │
│                                                             │
│   迁移过程 (每次操作迁移一个 bucket):                       │
│   ┌────────────────────┐  ┌────────────────────┐          │
│   │    ht[0]           │  │    ht[1]           │          │
│   │  [0]→迁移完成     │  │  [0]→a→b          │          │
│   │  [1]→c            │  │     空             │          │
│   └────────────────────┘  └────────────────────┘          │
│   rehashidx = 1                                            │
│                                                             │
│   迁移完成后: 交换 ht[0] 和 ht[1]                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.4 QuickList (List 实现)

```c
// src/quicklist.h
typedef struct quicklist {
    quicklistNode *head;
    quicklistNode *tail;
    unsigned long count;        // 总元素数
    unsigned long len;         // 节点数
    int fill : QL_FILL_BITS;  // 每个 ziplist 最大大小
    int compress : QL_COMP_BITS; // 压缩深度
    int bookmark_offset;        // 书签偏移
} quicklist;

typedef struct quicklistNode {
    struct quicklistNode *prev;
    struct quicklistNode *next;
    unsigned char *zl;         // 指向 ziplist
    unsigned int sz;           // ziplist 大小
    unsigned int count : 16;   // 元素数量
    unsigned int encoding : 2;  // RAW(1) or LZF(2)
    unsigned int container : 2; // NONE or ZIPLIST
    unsigned int recompress : 1; // 是否需要解压
    unsigned int attempted_compress : 1; // 尝试压缩标记
    unsigned int extra : 10;    // 额外字段
} quicklistNode;
```

---

### 2.5 Skiplist (跳表)

```c
// src/server.h
typedef struct zskiplist {
    struct zskiplistNode *header, *tail;
    unsigned long length;       // 节点数量
    int level;                 // 当前最大层数
} zskiplist;

typedef struct zskiplistNode {
    sds ele;                  // 成员
    double score;              // 分数
    struct zskiplistNode *backward;
    struct zskiplistLevel {
        struct zskiplistNode *forward;
        unsigned long span;   // 跨度
    } level[];
} zskiplistNode;
```

#### 跳表结构图

```
┌─────────────────────────────────────────────────────────────┐
│                      SkipList 结构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Level 3:  HEAD ───────────────────────────────────────►  NULL
│            │  │
│   Level 2:  HEAD ─────► [score:50] ────────────────────►  NULL
│            │         │
│   Level 1:  HEAD ─►[10]─►[30]─►[50]─►[80]─►[100]─────►  NULL
│                                                             │
│   span 记录每个节点到下一个节点的"跨越"元素数量               │
│   通过 span 可以 O(log n) 计算排名                         │
│                                                             │
│   查找过程:                                                 │
│   1. 从最高层开始，向右找到最后一个 <= 目标值的节点          │
│   2. 下降到下一层，重复                                      │
│   3. 到达最底层后，向右一步就是目标（或不存在）             │
│                                                             │
│   优点 vs 红黑树:                                           │
│   - 实现更简单                                              │
│   - 范围查询更高效 (只需遍历一层)                          │
│   - 插入/删除不需要旋转                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.6 ziplist (压缩列表)

```c
// src/ziplist.c
/*
 * ziplist 内存布局:
 *
 * <zlbytes> <zltail> <zllen> <entry> <entry> ... <zlend>
 *
 * zlbytes: 4 bytes - 总字节数
 * zltail:  4 bytes - 最后一个元素的偏移
 * zllen:   2 bytes - 元素数量
 * zlend:   1 byte  - 0xFF (结束标记)
 */

typedef struct zlentry {
    unsigned int prevrawlensize; // 前一个元素的长度字段大小
    unsigned int prevrawlen;    // 前一个元素的长度
    unsigned int lensize;       // 当前元素长度字段大小
    unsigned int len;           // 当前元素长度
    unsigned int headersize;    // 头大小
    unsigned char encoding;     // 编码方式
    unsigned char *p;           // 实际数据指针
} zlentry;
```

---

### 2.7 intset (整数集合)

```c
// src/intset.h
typedef struct intset {
    uint32_t encoding;      // 编码: INTSET_ENC_INT16/32/64
    uint32_t length;        // 元素数量
    int8_t contents[];      // 柔性数组存储元素
} intset;

// 编码常量
#define INTSET_ENC_INT16 (sizeof(int16_t))
#define INTSET_ENC_INT32 (sizeof(int32_t))
#define INTSET_ENC_INT64 (sizeof(int64_t))
```

---

## 三、事件循环 (aeEventLoop)

### 3.1 核心结构

```c
// src/ae.h
typedef struct aeEventLoop {
    int maxfd;                  // 最大文件描述符
    int setsize;                // fd 集合大小
    aeFileEvent *events;       // 文件事件数组
    aeFiredEvent *fired;       // 已触发事件数组
    aeTimeEvent *timeEventHead;// 时间事件链表
    int stop;                   // 停止标志
    void *apidata;              // 多路复用库数据 (epoll/select/kqueue)
    aeBeforeSleepProc *beforesleep;  // 睡眠前回调
} aeEventLoop;
```

### 3.2 主循环

```c
// src/ae.c
void aeMain(aeEventLoop *eventLoop) {
    eventLoop->stop = 0;
    while (!eventLoop->stop) {
        // 处理时间事件
        processTimeEvents(eventLoop);

        // 处理文件事件 (I/O多路复用)
        aeProcessEvents(eventLoop, AE_ALL_EVENTS|
                                   AE_CALL_BEFORE_SLEEP|
                                   AE_CALL_AFTER_SLEEP);
    }
}
```

---

## 四、内存管理 (jemalloc)

### 4.1 内存分配器

```
┌─────────────────────────────────────────────────────────────┐
│              Redis 内存管理 (jemalloc)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Redis 默认使用 jemalloc 作为内存分配器                    │
│                                                             │
│   优势:                                                     │
│   1. 内存碎片率低                                          │
│   2. 分配效率高 (slab allocator)                          │
│   3. 支持多线程 (减少锁竞争)                                │
│                                                             │
│   内存分配策略:                                            │
│   - 小块: 从 thread-local cache 分配                       │
│   - 大块: 从 central cache 分配                            │
│   - 释放: 归还到对应的 cache                               │
│                                                             │
│   查看内存使用:                                             │
│   > INFO memory                                            │
│   > MEMORY STATS                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、网络通信

### 5.1 协议解析 (RESP)

```c
// src/networking.c
/*
 * Redis 协议 (RESP - Redis Serialization Protocol)
 *
 * 简单字符串: +OK\r\n
 * 错误:       -ERR message\r\n
 * 整数:       :1000\r\n
 * 批量字符串: $6\r\nfoobar\r\n
 * 数组:       *2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n
 */
```

### 5.2 I/O 多路复用

```
┌─────────────────────────────────────────────────────────────┐
│                 Redis I/O 模型                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   主线程 (单线程):                                          │
│   ┌─────────────────────────────────────────────────┐      │
│   │  1. 等待 I/O (epoll/kqueue/select)              │      │
│   │  2. 处理已就绪的客户端连接                       │      │
│   │  3. 解析命令                                     │      │
│   │  4. 执行命令                                     │      │
│   │  5. 返回响应                                     │      │
│   └─────────────────────────────────────────────────┘      │
│                                                             │
│   瓶颈分析:                                                │
│   - 命令执行: O(1) 或 O(log n)，非常快                     │
│   - 网络 I/O: 主要瓶颈                                      │
│   - 解决方案: I/O Threads (Redis 6+)                      │
│                                                             │
│   Redis 6.0+ I/O Threads:                                 │
│   ┌──────────┬──────────┬──────────┐                      │
│   │ Main     │ IO       │ IO       │                      │
│   │ Thread   │ Thread 1 │ Thread 2 │                      │
│   │ (1)      │ (2)      │ (3)      │                      │
│   └──────────┴──────────┴──────────┘                      │
│   - 主线程: 命令解析 + 执行 + 响应写入                     │
│   - I/O 线程: 读取客户端数据 + 读取响应数据               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 六、持久化实现

### 6.1 RDB 快照

```c
// src/rdb.c
/*
 * RDB 文件结构:
 *
 * +------------------+---------------+
 * |  REDIS (5 bytes) |  版本号 (4)   |
 * +------------------+---------------+
 * |  AUX fields      |  数据库       |
 * +------------------+---------------+
 * |  KEY-VALUE PAIRS |  (多个)       |
 * +------------------+---------------+
 * |  EOF (0xFF)     |  CRC64        |
 * +------------------+---------------+
 */
```

### 6.2 AOF 重写

```
┌─────────────────────────────────────────────────────────────┐
│                    AOF 重写过程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   重写前:                                                   │
│   AOF: SET k1 v1, SET k1 v2, SET k1 v3, DEL k1            │
│                                                             │
│   重写后:                                                   │
│   AOF: (无操作，或只保留最终状态)                           │
│                                                             │
│   Redis 4.0+ 混合模式:                                     │
│   - AOF 重写时先用 RDB 格式快速压缩                        │
│   - 后续增量用 AOF 格式                                    │
│                                                             │
│   AOF 文件示例:                                            │
│   *3\r\n$3\r\nSET\r\n$2\r\nk1\r\n$2\r\nv3\r\n           │
│   (RESP 格式)                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 七、主从复制

```c
// src/replication.c

/*
 * 复制流程:
 *
 * 1. 主节点启动时创建 replid 和 offset
 * 2. 从节点连接主节点，发送 PSYNC replid offset
 * 3. 主节点判断:
 *    - 如果 offset 存在: 增量复制
 *    - 如果 offset 不存在: 全量复制 (RDB)
 * 4. 主节点持续发送命令 + 心跳
 */
```

---

## 八、关键函数调用链

### 8.1 处理客户端命令

```
┌─────────────────────────────────────────────────────────────┐
│                  命令处理调用链                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   main()                                                    │
│    └─► aeMain()                                            │
│         └─► aeProcessEvents()                              │
│              └─► acceptTcpHandler()                         │
│                   └─► createClient()                       │
│                        └─► readQueryFromClient()          │
│                             └─► processInputBuffer()        │
│                                  └─► processCommand()      │
│                                       └─► call()           │
│                                            └─► command->proc()
│                                             (如: setCommand, getCommand)
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 SET 命令执行流程

```
SET key value

1. processCommand() 解析命令
2. setCommand()     调用 t_string.c 中的 setCommand
3. setGenericCommand() 通用设置逻辑
4. dbAdd()          写入数据库
5. signalModifiedKey() 发送通知
6. rewriteClientCommandVector() 记录 AOF
7. addReply()       发送响应
```

---

## 九、Redis 6/7 多线程模型

### 9.1 I/O Threads

```c
// src/server.h
/* I/O threads configuration */
int io_threads_num;        // I/O 线程数量
int io_threads_do_reads;  // 是否让 I/O 线程读取数据

// 配置:
// io-threads 4
// io-threads-do-reads yes
```

```
┌─────────────────────────────────────────────────────────────┐
│              Redis 6/7 多线程模型                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   主线程 (Main Thread):                                    │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ 1. 监听端口，接受连接                                │  │
│   │ 2. 为客户端创建 fd                                  │  │
│   │ 3. 将 fd 分配给 I/O 线程                           │  │
│   │ 4. I/O 线程读取数据到 query buf                    │  │
│   │ 5. 主线程解析命令                                   │  │
│   │ 6. 主线程执行命令                                   │  │
│   │ 7. I/O 线程写回响应                                 │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   I/O 线程:                                                │
│   - 只负责读写 Socket                                       │
│   - 不执行命令                                              │
│   - 使用锁 + 环形队列                                      │
│                                                             │
│   配置示例:                                                 │
│   io-threads 4        # 4 个 I/O 线程                     │
│   io-threads-do-reads yes  # 启用多线程读取                │
│                                                             │
│   性能提升:                                                │
│   - 单核: ~30% 提升                                       │
│   - 多核: 线性提升                                         │
│   - 瓶颈: 命令执行 (仍单线程)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Redis 7.0+ 多线程改进

```
┌─────────────────────────────────────────────────────────────┐
│            Redis 7.0-8.x 多线程改进                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Redis 7.2 重大改进:                                      │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ 1. Client Output Buffer 线程化                       │  │
│   │    - 输出缓冲区写入独立线程                          │  │
│   │    - 减少主线程阻塞                                 │  │
│   │                                                     │  │
│   │ 2. 多个后台任务线程化                               │  │
│   │    - AOF 写入                                       │  │
│   │    - 惰性删除                                       │  │
│   │    - 统计信息收集                                   │  │
│   │                                                     │  │
│   │ 3. 主线程架构优化                                   │  │
│   │    - 减少全局锁竞争                                │  │
│   │    - 无锁数据结构改进                               │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   Redis 8.x 进一步优化:                                    │
│   - RFCH (Redesigned Fast Cursor Hash)                     │
│   - TSAN (NUMA 感知优化)                                  │
│   - 查询缓存                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 十、调试与诊断

### 10.1 常用调试命令

```bash
# 查看数据类型编码
> OBJECT ENCODING mykey
"ziplist"  # 或 "hashtable", "quicklist", "skiplist"

# 查看空闲时间
> OBJECT IDLETIME mykey

# 查看引用计数
> OBJECT REFCOUNT mykey

# 内存分析
> MEMORY DOCTOR
> MEMORY STATS
> MEMORY USAGE mykey

# 慢查询
> SLOWLOG GET 10
```

### 10.2 GDB 调试

```bash
# 编译调试版本
make CFLAGS="-g -O0"

# 启动调试
gdb src/redis-server

# 常用命令
(gdb) break setCommand    # 设置断点
(gdb) run                 # 运行
(gdb) bt                  # 查看调用栈
(gdb) p *obj              # 打印对象
```

---

## 十一、参考资料

- [Redis 官方源码](https://github.com/redis/redis)
- [Redis 设计与实现](https://github.com/huangz1990/redis-3.0-annotated)
- [Redis 源码注释](https://github.com/menwengit/RedisSourceAnnotation)
