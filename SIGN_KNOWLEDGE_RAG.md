# 手语词典 RAG 导入

## 数据隔离

旧集合 `SignKnowledgeChunks` 保留原样。本导入器只允许写入新集合 `SignKnowledgeChunks_v2`，不提供旧集合选项，也不删除任何集合。Web 应用默认仍使用旧集合；只有服务端显式配置 `BOOK_KNOWLEDGE_COLLECTION=SignKnowledgeChunks_v2` 才会切换，两个集合不会同时参与检索。回退时移除该配置并重启服务即可。

新解析数据和所有图片关联都先处于待审核状态，不会进入问答。解析文本变化时续跑会清空旧向量并重置为待审核；内容 hash 未变化时续跑会保留人工审核状态和有效向量。稳定 `_id` 使重复运行可以幂等更新。

## 本地解析和审核

将两册 EPUB、两册 PDF 放在项目根目录。只解析并生成审核报告，不连接 Atlas：

```powershell
.\.venv\Scripts\python.exe import_sign_books.py
```

报告在项目根目录的 `book_review_report_v2.json`，原图提取至 `book_media/`。报告包含每个词条的稳定 ID、原文摘录、出处、图片路径、解析置信度和待确认原因。请逐条检查：词条原文是否只属于标题对应的词条、编号是否保留、图片是否对应、PDF 页码/EPUB 锚点是否正确。报告中标为 pending 的内容不会参与检索。

当前 EPUB 解析沿 spine 阅读顺序读取 XHTML，以有 ID 的词条标题作为边界，并使用目录链接核对标题；目录导航本身不作为词条。

PDF 根据词条标题字号、坐标和连续阅读顺序切分，词条延续到下一个标题，包含下一页开头的续文和续图。合并同一行拆成不同文本块的中文/拼音标题及折行标题，保留短说明。附录和索引不作为词典动作词条。每条保留 `source_segments`、`image_evidence` 和 `page_end`，可追溯原页及图片坐标；嵌在说明中的小手形符号也保留为 `inline_symbol` 图片，不编造其文字。

图片一般按标题区域及跨页顺序关联。第四册有5处插图位于其标题上方，已经查看原页，例外记录在 `book_pdf_layout_overrides.json`。例外只有在文件名和源文件 SHA256 完全匹配时才应用，图片证据标记为 `reviewed_layout_exception`；不可把这个规则推广为“所有标题上方的图都属于下一条”。不明确的边界仍产生待审核提示。扫描 PDF 没有文本层时不会 OCR 或生成替代文字。

本次原621条 PDF 解析提示的逐条处理记录为 `book_audit/warning_decisions.json`，原报告备份为 `book_audit/original_621_report.json`；原页联系表及目视核对清单在 `book_audit/warning_sheets/` 和 `book_audit/visual_manifest.json`。结论见 `book_audit/审核结论.md`。这些是本地审核资料，不修改 Atlas 审核状态，不表示全部 EPUB/PDF 词条均已逐字批准。

可用下面命令重新进行本地结构与检索检查，不连接 Atlas 或模型，不生成 embedding：

```powershell
.\.venv\Scripts\python.exe audit_book_pdf.py --reparse --audit --validate
```

检查包括唯一词条 ID、图片存在与唯一坐标归属、目录/附录排除及真实解析数据的“豆腐①/②”检索。仅复查现有本地结果时使用 `--validate`。本地输出保持 pending；不能因提示降为0就批量批准全部词条。旧图片文件可以保留在本地，但不会自动被挂到新词条上。

## 暂存导入

### 先试11条已核对的样本

脚本 `import_sign_books_sample.py` 固定只选11条：第一册“豆腐①、豆腐②、豆浆、毒辣（残酷）、毒品、独裁（专制）、父亲（爸爸）、父亲节”，第四册“医院、英国、账户”。新增5条也已核对 PDF 原页；“父亲节”和“医院”包含跨页图文，“英国”和“账户”使用经原页核对的插图归属例外。重新解析两册 PDF 后，逐条校验已核对版本的 ID 和内容 hash；不匹配即停止，不能把改过的原文自动批准。本地原文预览输出到 `book_audit/sample_import_preview.json`。

```powershell
# 本地预览，不连接云端
.\.venv\Scripts\python.exe import_sign_books_sample.py
# 只上传这11条，并将其已核对版本设为可检索
.\.venv\Scripts\python.exe import_sign_books_sample.py --upload
```

`--upload` 表示执行这11条的小批上传及其审核状态更新。脚本只写 `SignKnowledgeChunks_v2`；尚无集合时会自动创建，不清空已有集合，不批准其他词条，也不修改应用配置。重复运行按稳定 ID 幂等更新；已导入前6条时，重跑会补上新增5条。只有这11条可以由这个脚本批准，而且审核更新同时匹配内容 hash。默认不调用 Embedding，精确词条查询即可检查图文展示。

系统和导入器共用同一索引准备逻辑：如果新集合已有 `(book, review_status)` 普通索引，即使名称为旧启动器创建的 `knowledge_book_review`，也直接复用，不再要求另建同结构的 `knowledge_book_review_v2`，不删除或重命名原索引。初始化失败时终端会显示 MongoDB 错误编号和说明（如85为索引配置冲突、18为认证失败、13为权限不足），不会输出驱动原始异常中的连接串或秘密。

导入完成后，在运行 Flask 的终端按 Ctrl+C 停止当前服务，然后执行这一条命令，以5001端口查看新集合的原文和图片：

```powershell
.\.venv\Scripts\python.exe start_sample_system.py
```

`start_sample_system.py` 启动 v2 语义检索和 Qwen 回答；查询内容会发送到已配置的 Embedding/回答模型服务。若只想检查原书摘录、不调用模型，可直接运行 `serve_sign_api.py --sample-books --port 5001`。增加了端口占用及运行锁检查，避免 Windows 上两个服务同时占用5001：发现占用只提示，不终止不明进程、不换端口。访问 `http://127.0.0.1:5001/api/knowledge/status`，确认 `knowledgeApiVersion` 为2、`collection` 为 `SignKnowledgeChunks_v2`，再搜索已审核词条。

若启动了新版但 `approvedCount` 为0，说明当前服务连接的新集合还没有可搜索的记录；仅运行导入脚本的默认本地预览不会上传。执行上面的 `import_sign_books_sample.py --upload`，成功后即可在已运行的新版系统搜索，不必再次导入全部书籍。若接口没有 `knowledgeApiVersion` 或 `collection` 字段，则仍连到了旧进程；前端会拒绝显示该版本的搜索结果，提示重新启动。浏览器需要刷新以加载新的前端代码。

如果还要为这11条测试真实向量（会把原文发送至现有 Embedding 服务），运行：

```powershell
.\.venv\Scripts\python.exe import_sign_books_sample.py --upload --embed
```

每次只发送一条，固定单并发；临时错误间隔3秒，最多重试3次，默认要求1024维。相同内容的已有向量会跳过。脚本会尝试建立 v2 向量索引，提交创建请求不代表已就绪；需在 Atlas 查看。

搜索顺序为词条精确匹配、原文关键词匹配、Atlas 向量相似检索。自然语言词条问题会移除已知提问前后缀，例如“和葡萄有关的手语”会先匹配“葡萄”。没有直接命中时，向量检索最多显示3条候选，默认过滤相似度低于 `BOOK_SIMILARITY_MIN_SCORE=0.62` 的结果，并标注“暂未找到该词汇，为您推荐以下相似词汇”。

测试后回退：停止 Flask，在同一终端移除这些临时覆盖后按原方式启动；不会删除 v2 数据。

```powershell
Remove-Item Env:BOOK_KNOWLEDGE_COLLECTION -ErrorAction SilentlyContinue
Remove-Item Env:BOOK_EMBEDDING_INDEX_NAME -ErrorAction SilentlyContinue
Remove-Item Env:BOOK_CONTENT_EXTERNAL_ALLOWED -ErrorAction SilentlyContinue
```

未显式配置集合时应用使用旧集合；若原先已在 `.env` 指定其他集合，移除终端覆盖后会按 `.env` 加载。

确认书籍许可允许把原文存到 Atlas 后，使用显式授权参数写入 v2。下面的命令不生成向量：

```powershell
.\.venv\Scripts\python.exe import_sign_books.py --upload --allow-atlas-upload
```

确认许可还允许把原文发送至 Embedding 服务后，可小批验证真实向量（模型/API 配置必须已存在）：

```powershell
.\.venv\Scripts\python.exe import_sign_books.py --upload --embed --embedding-limit 20 --embedding-batch-size 1 --embedding-workers 1 --allow-atlas-upload --allow-cloud-processing
```

验证文本、维度、向量和审核报告无误后，补齐剩余向量：

```powershell
.\.venv\Scripts\python.exe import_sign_books.py --upload --embed --embedding-batch-size 1 --embedding-workers 1 --allow-atlas-upload --allow-cloud-processing
```

`--embedding-batch-size` 支持 1、4、8。设为8表示一个请求包含8条，仍然只同时发送一个请求；实际模型是否接受8条需要真实调用验证。切换前先用 Ctrl+C 停止原导入任务，再用相同命令将该参数改为8，已有且内容未变的向量会跳过。最后不足8条时按实际数量发送。若模型拒绝该批量，改回4或1续跑。

命令只写 `SignKnowledgeChunks_v2`。Embedding 串行执行；失败时最多重试 3 次，每次间隔 3 秒。中断后可重跑；相同内容不重复生成已有向量。终端和日志不会记录原文或秘密。索引使用 `sign_knowledge_vector_v2`，向量维度必须以真实 Embedding 返回值为准。若自动建索引失败，在 Atlas Search 为 v2 集合建立 Vector Search 索引：

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine" },
    { "type": "filter", "path": "review_status" }
  ]
}
```

将 `numDimensions` 替换成服务实际返回的维度。索引名需为 `sign_knowledge_vector_v2`。向量缺失或索引未就绪时应用只能使用词条精确/关键词检索，不会伪造向量结果。

## 审核和切换

报告里的 `pending_chunks` 是需要人工检查的词条；`review_items` 是解析器遇到的问题。确认某条内容后，用其 ID 单独审核：

```powershell
.\.venv\Scripts\python.exe import_sign_books.py --approve CHUNK_ID
.\.venv\Scripts\python.exe import_sign_books.py --reject CHUNK_ID
```

审核命令也只连接 v2。不要因旧集合中曾经审核过，就自动批准新切分条目。全部确认并完成 v2 向量索引后，才在 Flask 服务环境显式设置：

```ini
BOOK_KNOWLEDGE_COLLECTION=SignKnowledgeChunks_v2
BOOK_EMBEDDING_INDEX_NAME=sign_knowledge_vector_v2
```

重启服务后验证搜索。回退时移除 `BOOK_KNOWLEDGE_COLLECTION`（默认旧集合）并重启。迁移期间不删除旧集合，也不把旧集合数据复制或混入 v2。

## Atlas 连接与状态

网页服务和导入器统一通过 `create_mongodb_client` 建立连接：默认使用 certifi CA 证书，也保留显式的 `MONGODB_TLS_CA_FILE` 配置。连接、服务器选择和读写均有超时限制；不会自动关闭证书校验。

`/api/knowledge/status` 仅在数据库查询成功时返回词条数量。连接失败返回503、`databaseAvailable=false` 和空数量，不再将读取失败伪装成“0条资料”。因此看到连接失败时应先恢复连接，无需据此重复上传书籍。

## 模型配置

Embedding 默认使用 `BOOK_EMBEDDING_BASE_URL`、`BOOK_EMBEDDING_MODEL` 和 `DASHSCOPE_API_KEY`（或项目已有的 DashScope Key 名称）；Qwen 回答模型读取 `BOOK_QWEN_BASE_URL`、`BOOK_QWEN_MODEL` 与配置的 Qwen/DashScope Key。凭据只放在本地 env 文件或环境变量中。没有真实凭据、真实 API 响应或有效向量索引时，不能宣称完成了云端 Embedding 或 Atlas Vector Search。
