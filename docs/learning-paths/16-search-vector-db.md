# 搜索引擎与向量数据库（2天）

## 概述
- **目标**：系统掌握搜索引擎技术与向量数据库的原理与应用，深入理解倒排索引、相关性排序、向量检索等核心技术，满足JD中"构建服务于搜索、多模态与模型训练的高质量数据湖与索引系统"的要求
- **时间**：春节第3周（2天）
- **前提**：熟悉基本数据结构（树、哈希表），了解机器学习基础概念
- **强度**：中高强度（每天8小时），适合需要构建搜索/向量检索系统的工程师

## JD要求对应

### JD领域覆盖
| JD领域 | 对应内容 | 优先级 |
|--------|----------|--------|
| 一、高并发服务端与API系统 | 搜索服务性能优化、查询引擎设计 | ⭐⭐⭐ |
| 二、大规模数据处理Pipeline | 数据索引构建、向量索引、搜索排序 | ⭐⭐⭐ |
| 三、Agent基础设施与运行时平台 | RAG检索、知识库搜索 | ⭐⭐ |
| 四、异构超算基础设施 | 向量检索加速、GPU向量计算 | ⭐⭐ |

### JD能力对应
| 能力要求 | 学习内容 | 验证方式 |
|----------|----------|----------|
| **搜索引擎技术** | 倒排索引、分词、相关性排序 | 搜索引擎实现 |
| **向量数据库** | 向量索引、相似度检索、嵌入模型 | 向量检索系统 |
| **数据湖索引** | 数据索引架构、列式存储、查询优化 | 索引设计方案 |
| **RAG检索增强** | 知识库构建、向量检索、上下文融合 | RAG系统实现 |

## 学习重点

### 1. 搜索引引擎基础（第1天上午）
**JD引用**："构建服务于搜索、多模态与模型训练的高质量数据湖与索引系统"

**核心内容**：
- 搜索引擎架构
  - **爬虫系统**（Crawler）：网页抓取、URL去重
  - **索引系统**（Indexer）：文档解析、倒排索引
  - **查询系统**（Query Processor）：查询解析、相关性排序
  - **排序系统**（Ranking）：多特征排序、学习排序
- 倒排索引原理
  - **正排索引**：文档ID → 词项列表
  - **倒排索引**：词项 → 文档列表
  - **词典（Dictionary）**：词项存储与查询
  - **倒排列表（Posting List）**：词项出现位置
  - **压缩存储**：Variable Byte、Elias Gamma
- 文档处理流程
  - **分词（Tokenization）**：中文分词（结巴分词、HanLP）
  - **词法分析（Lemmatization）**：词干提取
  - **停用词过滤（Stop Words）**：高频词过滤
  - **词项归一化**：大小写转换、同义词处理

**实践任务**：
- 实现简单的倒排索引
- 使用结巴分词进行中文分词
- 对比正排索引与倒排索引
- 构建小型文档索引

### 2. 相关性排序算法（第1天下午）
**JD引用**："负责核心服务的性能优化、数据库调优与分布式系统可靠性保障"

**核心内容**：
- 基础排序算法
  - **TF-IDF**（词频-逆文档频率）
    - TF：词项在文档中的频率
    - IDF：逆文档频率
    - TF-IDF公式计算
  - **BM25**（Okapi BM25）
    - TF饱和函数
    - 文档长度归一化
    - BM25公式
  - **向量空间模型（VSM）**
    - 文档向量化
    - 余弦相似度计算
- 排序学习（Learning to Rank）
  - **Pointwise方法**：分类/回归视角
    - 训练分类器预测相关/不相关
    - 训练回归模型预测相关性分数
    - 算法：逻辑回归、GBDT
  - **Pairwise方法**：比较视角
    - 学习文档对的相对顺序
    - 关注查询下文档排序
    - 算法：RankNet、LambdaRank
  - **Listwise方法**：列表视角
    - 直接优化排序列表
    - 考虑整体评价指标
    - 算法：LambdaMART、ListNet
- 排序特征
  - **文本特征**：TF-IDF、BM25、语言模型
  - **文档特征**：PageRank、内容新鲜度
  - **用户特征**：点击历史、个性化
  - **上下文特征**：时间、设备、位置

**实践任务**：
- 实现TF-IDF排序
- 实现BM25排序
- 使用LightGBM训练排序模型
- 对比不同排序算法效果

### 3. 向量检索基础（第1天晚上）
**JD引用**："构建服务于搜索、多模态与模型训练的高质量数据湖与索引系统"

**核心内容**：
- 向量表示与嵌入
  - **词向量（Word2Vec、GloVe）**
    - 分布式表示原理
    - 上下文窗口
    - 静态词向量
  - **上下文词向量（BERT、GPT）**
    - Transformer架构
    - 动态上下文表示
    - 微调与冻结
  - **句子向量（Sentence-BERT、InstructEmbed）**
    - 句子级嵌入
    - 语义相似度
  - **多模态嵌入（CLIP、BLIP）**
    - 图像-文本对齐
    - 跨模态检索
- 向量相似度
  - **欧氏距离（L2）**
  - **余弦相似度（Cosine）**
  - **点积（Dot Product）**
  - **曼哈顿距离**
  - **汉明距离**
- 向量索引分类
  - **精确检索**：暴力搜索、KD-Tree
  - **近似最近邻（ANN）**
    - 树结构索引（KD-Tree、Ball Tree）
    - 哈希索引（LSH局部敏感哈希）
    - 图索引（HNSW、NSG）
    - 量化索引（IVF、PQ、OPQ）

**实践任务**：
- 使用Sentence-BERT生成句子向量
- 实现HNSW向量检索
- 对比不同索引性能
- 构建向量检索Demo

### 4. 倒排索引深入（第2天上午）
**JD引用**："负责数据采集、清洗、去重与质量评估系统的设计与开发"

**核心内容**：
- 索引构建
  - **单机索引构建**：内存排序、外部归并
  - **分布式索引**：MapReduce索引构建
  - **实时索引**：内存索引 + 持久化
  - **增量索引**：变更日志 + 合并
- 索引更新
  - **全量重建**：定期重建索引
  - **增量更新**：小批量实时更新
  - **删除处理**：墓碑标记 + 合并清理
  - **版本管理**：多版本并发
- 索引优化
  - **分片策略**：按文档ID、哈希、地理
  - **副本策略**：读写分离、高可用
  - **压缩优化**：词典压缩、列表压缩
  - **缓存策略**：Result Cache、Filter Cache
- 查询处理
  - **查询解析**：语法树构建
  - **查询重写**：同义词扩展、拼写纠正
  - **查询执行**：倒排列表合并
  - **结果缓存**：LRU缓存、查询缓存

**实践任务**：
- 实现分布式索引构建
- 配置索引分片与副本
- 优化倒排列表压缩
- 实现查询重写

### 5. 全文检索系统（第2天下午）
**JD引用**："面向数千万日活用户的产品后端架构设计"

**核心内容**：
- Elasticsearch实战
  - **集群架构**：Master、Data、Coordinating节点
  - **索引设计**：Mapping、Settings、分片数
  - **查询语法**：Query DSL、聚合查询
  - **性能优化**：Refresh、Flush、Merge
- 分词与分析
  - **分析器（Analyzer）**
    - Character Filter（字符过滤）
    - Tokenizer（分词器）
    - Token Filter（词项过滤）
  - **中文分词器**
    - IK Analyzer
    - HanLP Analyzer
    - Jieba Analyzer
    - 自定义分词
- 高级查询
  - **全文查询**：Match、Match Phrase
  - **精确查询**：Term、Terms
  - **复合查询**：Bool、Function Score
  - **范围查询**：Range
  - **地理查询**：Geo Distance、Geo Polygon
- 聚合与分析
  - **指标聚合**：Sum、Avg、Percentiles
  - **桶聚合**：Terms、Date Histogram
  - **管道聚合**：移动平均、导数

**实践任务**：
- 设计Elasticsearch索引Mapping
- 配置中文分词器
- 实现复杂查询
- 使用聚合进行数据分析

### 6. 向量数据库深入（第2天下午）
**JD引用**："构建服务于搜索、多模态与模型训练的高质量数据湖与索引系统"

**核心内容**：
- 向量数据库架构
  - **存储层**：向量存储、元数据存储
  - **索引层**：向量索引、标量索引
  - **查询层**：向量检索、混合检索
  - **服务层**：API、负载均衡
- 向量索引算法
  - **HNSW（Hierarchical Navigable Small World）**
    - 分层图结构
    - 构建过程：入口点、贪心搜索
    - 参数：efConstruction、M、NN
  - **IVF（Inverted File Index）**
    - 聚类分桶
    - 倒排索引结构
    - 参数：nlist、nprobe
  - **PQ（Product Quantization）**
    - 向量分块
    - 聚类编码
    - 压缩存储
  - **OPQ（Optimized Product Quantization）**
    - 旋转优化
    - 均方误差优化
- 混合检索
  - **向量 + 关键词**：BM25 + 向量
  - **重排序（Re-ranking）**：两阶段检索
  - **稀疏向量**：Sparse Embedding
  - **RAG检索**：向量 + 知识图谱
- 生产实践
  - **性能调优**：召回率 vs 延迟
  - **规模化**：分片、复制、分布式
  - **一致性**：强一致 vs 最终一致
  - **运维**：监控、备份、恢复

**实践任务**：
- 使用Milvus构建向量数据库
- 配置HNSW索引参数
- 实现混合检索
- 对比不同向量数据库

### 7. RAG检索增强生成（第2天晚上）
**JD引用**："开发与迭代 AI Chat Bot 等创新产品功能"

**核心内容**：
- RAG架构
  - **检索（Retrieval）**：向量检索 + 关键词检索
  - **增强（Augmentation）**：上下文融合
  - **生成（Generation）**：大语言模型
- 知识库构建
  - **文档处理**：PDF解析、网页抓取
  - **分块策略**：固定窗口、重叠分块
  - **元数据提取**：标题、作者、时间
  - **质量评估**：相关性、准确性
- RAG优化
  - **查询改写**：HyDE、多查询
  - **重排序**：Cross-Encoder重排
  - **上下文压缩**：摘要、过滤
  - **检索融合**：多路召回
- RAG评估
  - **检索指标**：Hit Rate、MRR、NDCG
  - **生成指标**：BLEU、ROUGE
  - **端到端评估**：人工评估、A/B测试

**实践任务**：
- 构建RAG知识库
- 实现RAG检索
- 优化RAG效果
- 评估RAG性能

## 实践项目：RAG检索增强系统

### 项目目标
**JD对应**：满足"开发与迭代 AI Chat Bot 等创新产品功能"和"构建服务于搜索、多模态与模型训练的高质量数据湖与索引系统"要求

实现一个生产级RAG检索增强系统，包含：
1. 文档处理与向量化
2. 混合检索（向量 + 关键词）
3. 重排序机制
4. RAG上下文融合

### 技术栈参考（明确版本）
- **搜索引擎**：Elasticsearch 8.10+ / OpenSearch 2.11+
- **向量数据库**：Milvus 2.3+ / Qdrant 1.7+ / Weaviate 1.22+
- **嵌入模型**：Sentence-Transformers 2.2+ / OpenAI Embeddings
- **分词器**：HanLP / Jieba / IK Analyzer
- **RAG框架**：LangChain 0.1+ / LlamaIndex 0.9+
- **LLM**：OpenAI API / Claude API / 本地模型

### 环境配置要求
- **操作系统**：Linux（推荐Ubuntu 22.04）
- **Docker**：24.0+
- **依赖安装**：
  ```bash
  # 启动Milvus
  docker-compose -f milvus.yml up -d

  # 启动Elasticsearch
  docker-compose -f elasticsearch.yml up -d

  # 安装Python依赖
  pip install sentence-transformers langchain pymilvus elasticsearch
  ```

### 架构设计
```
rag-system/
├── document/                  # 文档处理
│   ├── loaders/              # 文档加载器
│   │   ├── pdf_loader.py     # PDF加载
│   │   ├── html_loader.py   # 网页加载
│   │   └── txt_loader.py    # 文本加载
│   ├── splitters/            # 文档分块
│   │   ├── recursive_splitter.py  # 递归分块
│   │   └── semantic_splitter.py   # 语义分块
│   ├── chunkers/             # 分块策略
│   │   ├── fixed_chunker.py  # 固定大小
│   │   └── overlap_chunker.py # 重叠分块
│   └── embeddings/            # 向量嵌入
│       ├── sentence_transformer.py
│       └── openai_embedder.py
├── index/                     # 索引构建
│   ├── vector_index/         # 向量索引
│   │   ├── milvus_index.py   # Milvus索引
│   │   └── qdrant_index.py   # Qdrant索引
│   ├── text_index/          # 文本索引
│   │   └── elastic_index.py  # ES索引
│   └── hybrid_index/         # 混合索引
│       └── hybrid_retriever.py
├── retrieval/                 # 检索模块
│   ├── vector_retriever.py   # 向量检索
│   ├── text_retriever.py     # 关键词检索
│   ├── hybrid_retriever.py   # 混合检索
│   └── reranker.py          # 重排序
├── generation/               # 生成模块
│   ├── llm.py               # LLM接口
│   ├── prompt.py             # 提示词模板
│   └── rag_chain.py         # RAG链
├── evaluation/               # 评估模块
│   ├── metrics.py           # 检索指标
│   └── evaluator.py        # 评估器
└── api/                      # API服务
    ├── main.py              # FastAPI服务
    ├── models.py            # 数据模型
    └── schemas.py           # 请求响应
```

### 核心组件设计

#### 1. 文档分块
```python
# chunkers/recursive_splitter.py
class RecursiveSplitter:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitters = [
            ("\n\n", 2),   # 段落
            ("\n", 1),     # 换行
            ("。", 1),     # 句子（中文）
            ("，", 1),     # 逗号（中文）
            (" ", 1),      # 空格
            ("", 1),       # 字符
        ]

    def split(self, text: str) -> List[str]:
        chunks = []
        start_idx = 0

        while start_idx < len(text):
            # 尝试按层级分割
            for separator, weight in self.splitters:
                if start_idx + self.chunk_size > len(text):
                    # 剩余部分直接作为一个chunk
                    chunks.append(text[start_idx:])
                    start_idx = len(text)
                    break

                # 在chunk_size范围内找最后分隔符
                chunk = text[start_idx:start_idx + self.chunk_size]
                last_sep = -1
                for sep, _ in self.splitters:
                    pos = chunk.rfind(sep)
                    if pos > last_sep:
                        last_sep = pos

                if last_sep > 0:
                    # 在分隔符处分割
                    end_idx = start_idx + last_sep + len(separator)
                    chunks.append(text[start_idx:end_idx])
                    start_idx = end_idx - self.chunk_overlap
                    break
            else:
                # 没有找到分隔符，按固定大小分割
                chunks.append(text[start_idx:start_idx + self.chunk_size])
                start_idx += self.chunk_size - self.chunk_overlap

        return chunks
```

#### 2. 混合检索
```python
# retrieval/hybrid_retriever.py
class HybridRetriever:
    def __init__(
        self,
        vector_store: Milvus,
        text_store: Elasticsearch,
        reranker: CrossEncoderReranker,
        weights: List[float] = [0.5, 0.5]
    ):
        self.vector_retriever = VectorRetriever(vector_store)
        self.text_retriever = TextRetriever(text_store)
        self.reranker = reranker
        self.weights = weights

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        fusion_method: str = "rrf"
    ) -> List[Document]:
        # 1. 向量检索
        vector_results = self.vector_retriever.search(
            query=query,
            top_k=top_k
        )

        # 2. 关键词检索
        text_results = self.text_retriever.search(
            query=query,
            top_k=top_k
        )

        # 3. 融合排序（RRF）
        fused_scores = self._rrf_fusion(
            vector_results,
            text_results,
            self.weights
        )

        # 4. 重排序
        reranked = self.reranker.rerank(
            query=query,
            documents=fused_scores[:top_k * 2]
        )

        return reranked[:top_k]

    def _rrf_fusion(
        self,
        vector_results: List[Document],
        text_results: List[Document],
        weights: List[float]
    ) -> List[Document]:
        # RRF公式：1 / (rank + k)
        k = 60  # RRF常数

        # 构建文档得分
        doc_scores = {}

        # 向量检索得分
        for rank, doc in enumerate(vector_results):
            score = (1 - weights[0]) * (1.0 / (rank + 1 + k))
            doc_scores[doc.id] = (doc, score)

        # 关键词检索得分
        for rank, doc in enumerate(text_results):
            if doc.id in doc_scores:
                old_doc, old_score = doc_scores[doc.id]
                doc_scores[doc.id] = (old_doc, old_score + weights[1] * (1.0 / (rank + 1 + k)))
            else:
                doc_scores[doc.id] = (doc, weights[1] * (1.0 / (rank + 1 + k)))

        # 按得分排序
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in sorted_docs]
```

#### 3. 重排序
```python
# retrieval/reranker.py
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        if not documents:
            return []

        # 构建查询-文档对
        pairs = [(query, doc.content) for doc in documents]

        # 计算相关性分数
        scores = self.model.predict(pairs)

        # 按分数排序
        doc_with_scores = list(zip(documents, scores))
        doc_with_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回top_k
        return [doc for doc, score in doc_with_scores[:top_k]]
```

## 学习资源

### 经典书籍
1. **《信息检索导论》**：Christopher Manning - IR经典教材
2. **《搜索引擎：信息检索实践》**：W. Bruce Croft - 实战指南
3. **《深度学习搜索与推荐》**：搜索推荐系统
4. **《向量数据库》**：Morgan Claypool - 向量检索指南

### 官方文档
1. **Elasticsearch官方文档**：[elastic.co/docs](https://www.elastic.co/docs)
2. **Milvus官方文档**：[milvus.io/docs](https://milvus.io/docs)
3. **HNSW论文**：[arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)
4. **BM25论文**：Stephen Robertson的BM25论文

### 在线课程
1. **Stanford CS276**：[信息检索](https://web.stanford.edu/class/cs276/)
2. **CMU 11-442**：[搜索引擎](https://www.cs.cmu.edu/~./sebastian/miracle.html)
3. **DeepLearning.AI**：[搜索推荐](https://www.deeplearning.ai/)

### 技术博客与案例
1. **Elastic Blog**：[搜索技术](https://www.elastic.co/blog)
2. **Milvus Blog**：[向量数据库](https://milvus.io/blog)
3. **Pinecone Blog**：[向量检索](https://www.pinecone.io/blog/)
4. **Netflix Tech Blog**：[搜索架构](https://netflixtechblog.com/)

### 开源项目参考
1. **Elasticsearch**：[github.com/elastic/elasticsearch](https://github.com/elastic/elasticsearch)
2. **Milvus**：[github.com/milvus-io/milvus](https://github.com/milvus-io/milvus)
3. **MeiliSearch**：[github.com/meilisearch/MeiliSearch](https://github.com/meilisearch/MeiliSearch)
4. **LanceDB**：[github.com/lancedb/lancedb](https://github.com/lancedb/lancedb)
5. **LangChain**：[github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

### 权威论文
1. **BM25**：[Okapi BM25](https://dl.acm.org/doi/10.1145/253212.253222)
2. **HNSW**：[Hierarchical Navigable Small World](https://arxiv.org/abs/1603.09320)
3. **FAISS**：[Facebook AI Similarity Search](https://arxiv.org/abs/1702.08734)
4. **BERT for IR**：[BERT-based IR](https://arxiv.org/abs/1910.10683)
5. **ColBERT**：[Contextualized Late Interaction](https://arxiv.org/abs/2112.01488)

### 实用工具
1. **Elasticsearch工具**：Kibana、Cerebro
2. **向量索引工具**：Faiss、Annoy
3. **嵌入工具**：Sentence-Transformers、HuggingFace
4. **RAG评估**：RAGAS、Trulens

## 学习产出要求

### 设计产出
1. ✅ 搜索引擎架构设计文档
2. ✅ 向量数据库选型方案
3. ✅ RAG系统架构设计
4. ✅ 混合检索方案

### 代码产出
1. ✅ 倒排索引实现
2. ✅ 向量检索系统（Milvus）
3. ✅ RAG检索链实现
4. ✅ 混合检索融合

### 技能验证
1. ✅ 理解倒排索引原理
2. ✅ 掌握相关性排序算法
3. ✅ 能够构建向量索引
4. ✅ 能够实现混合检索
5. ✅ 能够设计RAG系统

### 文档产出
1. ✅ 搜索引擎技术选型报告
2. ✅ 向量数据库对比评测
3. ✅ RAG系统设计文档

## 时间安排建议

### 第1天（搜索与向量基础）
- **上午（4小时）**：搜索引擎基础
  - 倒排索引原理
  - 相关性排序
  - 实践：实现倒排索引

- **下午（4小时）**：向量检索
  - 嵌入模型原理
  - 向量索引算法
  - 实践：使用HNSW

- **晚上（2小时）**：Elasticsearch
  - 索引设计
  - 查询语法
  - 实践：构建搜索服务

### 第2天（RAG与实践）
- **上午（4小时）**：向量数据库
  - Milvus/Qdrant
  - 混合检索
  - 实践：构建向量检索

- **下午（4小时）**：RAG系统
  - 知识库构建
  - 检索增强
  - 实践：实现RAG链

- **晚上（2小时）**：总结
  - 系统集成
  - 性能优化
  - 制定后续计划

## 学习方法建议

### 1. 理论与实践结合
- 理解倒排索引原理
- 实现简单索引
- 使用Elasticsearch实战
- 构建RAG系统

### 2. 关注性能指标
- 搜索引擎：QPS、延迟、召回率
- 向量检索：召回率、延迟、内存
- RAG系统：答案质量、延迟

### 3. 从搜索到向量
- 先理解传统搜索
- 再学习向量检索
- 最后掌握混合检索

## 常见问题与解决方案

### Q1：如何选择向量数据库？
**A**：选型考虑：
- **Milvus**：功能全、社区活跃
- **Qdrant**：Rust实现、性能好
- **Pinecone**：托管服务、云原生
- **Weaviate**：知识图谱结合

### Q2：如何优化向量召回率？
**A**：优化方向：
- **索引参数**：增大HNSW的efSearch
- **嵌入质量**：选择合适的嵌入模型
- **分块策略**：合理分块大小
- **混合检索**：结合关键词检索

### Q3：如何优化搜索延迟？
**A**：优化方向：
- **缓存**：Result Cache、Query Cache
- **预计算**：缓存排序分数
- **异步**：并行检索、融合
- **索引优化**：分片、压缩

### Q4：RAG如何提升质量？
**A**：提升方向：
- **分块优化**：语义分块、重叠
- **查询改写**：HyDE、多查询
- **重排序**：Cross-Encoder
- **上下文压缩**：摘要、过滤

### Q5：如何评估RAG系统？
**A**：评估指标：
- **检索**：Hit Rate、MRR、NDCG
- **生成**：BLEU、ROUGE
- **端到端**：人工评估、A/B测试
- **工具**：RAGAS、Trulens

## 知识体系构建

### 核心知识领域

#### 1. 搜索引擎
```
搜索引擎
├── 爬虫系统
│   ├── URL去重
│   ├── 页面抓取
│   └── 内容解析
├── 索引系统
│   ├── 倒排索引
│   ├── 正排索引
│   └── 索引压缩
├── 查询处理
│   ├── 查询解析
│   ├── 查询重写
│   └── 结果排序
└── 排序算法
    ├── TF-IDF
    ├── BM25
    └── 学习排序
```

#### 2. 向量检索
```
向量检索
├── 向量嵌入
│   ├── 词向量
│   ├── 句子向量
│   └── 多模态嵌入
├── 向量索引
    ├── 精确检索
    ├── 近似检索（ANN）
    │   ├── 树结构（KD-Tree）
    │   ├── 哈希（LSH）
    │   ├── 图（HNSW、NSG）
    │   └── 量化（IVF、PQ）
    └── 混合索引
└── 相似度计算
    ├── 欧氏距离
    ├── 余弦相似度
    └── 点积
```

#### 3. RAG系统
```
RAG系统
├── 文档处理
│   ├── 文档加载
│   ├── 文档分块
│   └── 向量化
├── 检索模块
│   ├── 向量检索
│   ├── 关键词检索
│   └── 混合检索
├── 重排序
│   ├── Cross-Encoder
│   └── Listwise排序
└── 生成模块
    ├── 上下文融合
    └── 提示词工程
```

### 学习深度建议

#### 精通级别
- 倒排索引原理与实现
- BM25/TF-IDF排序
- HNSW/IVF索引
- RAG系统设计

#### 掌握级别
- Elasticsearch使用
- 向量数据库运维
- 嵌入模型选择
- 重排序技术

#### 了解级别
- 分布式索引构建
- 学习排序模型
- 多模态检索
- 图检索技术

## 下一步学习

### 立即进入
1. **数据工程**（路径07）：
   - 数据湖架构
   - 数据管道设计
   - 协同效应：搜索 + 数据工程

2. **AI/ML系统性学习**（路径04）：
   - 嵌入模型原理
   - 大语言模型应用

### 后续深入
1. **云原生进阶**（路径06）：搜索服务容器化
2. **实践项目**：智能问答系统

### 持续跟进
- 向量数据库新发展
- 多模态检索技术
- RAG评估标准演进

---

## 学习路径特点

### 针对人群
- 需要构建搜索/向量检索系统的工程师
- 面向JD中的"搜索与索引"要求
- 适合需要开发AI Chat Bot的工程师

### 学习策略
- **中高强度**：2天集中学习，每天8小时
- **理论与实践结合**：原理 + 实现
- **RAG导向**：聚焦AI应用场景

### 协同学习
- 与数据工程路径：数据管道
- 与AI/ML路径：嵌入模型
- 与高性能API路径：搜索服务

### 质量保证
- 技术栈版本明确
- 代码示例可直接运行
- 项目架构完整

---

*学习路径设计：针对需要构建搜索与向量检索系统的工程师*
*时间窗口：春节第3周2天，中高强度学习搜索引擎与向量数据库*
*JD对标：满足JD中搜索、索引、RAG等核心要求*
