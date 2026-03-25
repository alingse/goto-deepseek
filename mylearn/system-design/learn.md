1. api-gateway
   1. 考虑阅读更多 BFF
2. 缓存
   1. 分布式共享缓存
   2. 内存缓存，内存缓存如何更新
   3. 如何防止穿透, 限流?
      1. 缓存丢失
   4. 在 LLM 里面的 缓存怎么看待呢？
   5. 如何失效缓存呢？
      1. 内存缓存如何失效
      2. 一致性要求?
   6. single-fight 如何做，分布式 single-fight 如何做
   7. Redis 序列化的问题?
   8. Redis 大key
3. CAP
   1. 概念
      1. C ---> 一致性
      2. A ---> 可用性
      3. P ---> 分区容错性
   2. 相关问题
      1. 投票？
      2. 分布式协议？
4. CDN
   1. 边缘设备有什么成本
   2. 一致性如何保障？
   3. 网络联通怎么处理
5. 断路器
   1. 限流如何做
   2. 服务端限流 or 客户端限流？
6. CLUSTER
   1. 负载均衡
   2. 节点1 和节点2 怎么分配流量？
      1. round robin
   3. 主/ 从 模式？
7. 一致性 hash
   1. redis 缓存的应用
   2. 是否有其他类似的环形结构?
      1. Redis 队列？
      2. buffer ？
      3. go 的 channel
      4. io_ring ?
   3. 取模
8. 读写分离的模式
   1. 考虑？追问？
   2. 一致性？
9. 数据库副本
   1.  Tidb 副本
       1.  Tikv 和 pd 相关
   2.  MySQL 实例
   3.  主从架构？
   4.  Redis 呢？
   5.  读写分离客户端？
   6.  读的分片怎么做
10. 数据库
    1. 关系型数据库
    2. 文档型数据库
    3. 列数据库
    4. 图数据库
11. 故障恢复
    1.  Downtime 如何处理
    2.  downtime 流程/复盘/改进措施
12. 分布式事务
    1. 两阶段提交 ---> 点解?
    2. 三阶段提交 ---> 多了什么优势呢？
    3. 对比通过消息和事件传递的呢？
13. DNS
    1.  domain name system
    2.  拉取式更新，一层层往上
    3.  更多复杂问题？
14. enterprise-service-bus ？
    1.  这是什么?
15. 事件驱动设计
    1.  Event Router
    2.  发送者，消费者
    3.  事件存储 +Replay
16. geohashing
    1.  瓦片
17. 索引Index
    1.  MySQL 索引
    2.  Redis Hash
    3.  elesticsearch
    4.  mongodb
18. 负载均衡
    1.  网关
    2.  提升质量
    3.  网关错误统计
    4.  指标驱动
    5.  如何测量
19. Websocket
    1.  https wss
    2.  http 1.1 upgrade 消息握手
    3.  双通道
    4.  关闭重连
    5.  网关 websocket 长链接
20. message broker
    1.  Kafka 相关
    2.  订阅 topic
    3.  分区
    4.  消费者
21. 消息队列
22. 单体架构
23. 微服务架构
    1.  如何防止服务爆炸
    2.  最好不要分层太多
    3.  要有一定的垂直
24. N 层架构
25. netflix
    1. 视频的处理
    2. 用户 -> ISP -> Load Balance -> search -> user/video -> cache /db
    3. stream 服务
       1. media
    4. CDN
26. oauth
   1. oauth 2.0
27. network
    1.  OSI 7层架构
    2. 物理层 --> 光缆
    3. 链路层 -->
    4. 网络层 --> IP
    5. 传输层 --> TCP/等等协议
    6. 会话层
    7. 持久层
    8. 应用层 ---> 应用
28. proxy
    1.  Nginx
    2.  前向 proxy (客户端做分发)
    3.  反向代理
        1.  nginx --> 分发到不同的服务
29. 发布-订阅
    1.  Topic / router
    2.  最少一次
    3.  最多一次
30. 限流
    1.  限流器的设计，按用户？按地域？如何保护
31. scale
    1.  横向扩展 vs 垂直扩展
32. 服务发现
    1.  DNS ？
    2.  consul
    3. 服务注册
    4. LB 来承担流量
    5. mesh 怎么考虑
33. 分片
    1.  数据库分片，按照什么分
        1.  比如 chatdb 按照什么来分
    2.  比如用户私信，按照什么来分
34. 冷热数据分离
35. SSO
    1.  原理
    2.  oauth2.0
    3.  如何退出
    4.  jwt ? cookie / beaber
36. 经典的 TCP/UDP
    1.  tcp 三段握手
        1.  SYN | SYN+ACK | ACK +DATA
    2.  udp 重发
37. 事务
    1. 部分提交
    2. 已提交
    3. MySQL 的事务
38. twitter
    1. 社交网站类的设计
       1. 推荐流
       2. 关注流（拉模式
    2. newsfeed 的设计
    3. user -> LB -> tweet-service -> LB -> DB+cache
    4. search
    5. CDN
39. uber 的设计
    1. 打车 -> 寻找司机 -> 司机接受 -> 行驶 -> 完成订单
40. url 短链的设计
41. 消息App的设计
