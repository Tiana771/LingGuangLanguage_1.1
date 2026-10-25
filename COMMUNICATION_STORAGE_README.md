# 沟通记录、账号与检索配置

## Atlas 保存

Flask 使用项目虚拟环境中的 PyMongo 同步驱动。连接配置从进程环境变量或项目根目录的 `atlasconnection.env`、`.env`、旧文件名中读取。文件可包含原始 `mongodb+srv://...` URI，也可使用 `MONGO_URI=...`；若 URI 没有数据库路径，需另设 `MONGODB_DATABASE`。不要把任何环境文件提交到 Git。

启动或首次访问账号/沟通 API 时，应用会 ping 指定数据库，并确保以下集合和索引存在，不会清空数据库：

- `users`：用户名规范化键唯一索引，密码使用 Werkzeug 安全哈希。
- `LingGuangChat`：沟通记录，查询索引为 `owner_user_id, updatedAt`。
- `LingGuangAssistantChats`：该账号自己的历史问答。
- `sessions`：服务端会话，按到期时间 TTL 清理。

已登录的沟通记录只从 Atlas 读取和写入；旧版浏览器 IndexedDB 不迁移，也不会合并进登录账号的记录。手语识别与手语视频查询仍可免登录。

## TLS 与本机测试

正常配置应使用系统证书链校验 TLS。可选的 `MONGODB_TLS_CA_FILE` 用于指定可信 CA 文件。仅当本机 Python 环境确实遇到证书链问题时，开发者可在本机环境配置中显式设置 `MONGODB_TLS_ALLOW_INVALID_CERTIFICATES=true` 进行临时比赛演示测试。此选项会关闭服务器证书校验，存在中间人风险；不要用于生产或公共网络部署，也不要把它设为默认值。

`connecttest.py` 只执行连接与索引初始化，结果不会打印 URI、用户名、密码或异常消息。连接失败时请检查连接文件是否存在、Atlas Database User、Network Access 白名单和本机 TLS 配置。

## DeepSeek

选中文字总结、回复建议和历史记录问答都需要真实的 DeepSeek API 配置。应用读取 `DEEP_SEEK_API_KEY`，可选 `DEEP_SEEK_BASE_URL`。缺少密钥时接口会返回未配置错误，不会生成模拟结果。密钥只放在本机环境配置中，不要提交或打印。

## 历史记录检索现状

“问我的沟通记录”目前使用服务端 MongoDB 文本/关键词匹配降级：搜索时 Mongo 查询首先限制 `owner_user_id`，返回命中语音框的原文及沟通、主题、时间信息，再将这些命中内容交给 DeepSeek。没有命中时会明确告知未找到记录。页面会显示当前为关键词检索。该方式不等同于语义向量检索，无法保证理解同义表达；DeepSeek 也只会依据提供的检索片段回答。

### 启用 Atlas Vector Search 前仍需提供的真实配置

当前代码和环境未发现可用 Embedding 模型凭据、模型标识或 Atlas Vector Search 索引，因此没有创建向量、向量索引或伪造向量命中。要切换到向量检索，需先选定并真实开通 Embedding 服务，然后在 Atlas 为专用语音框向量集合建立 Vector Search 索引。建议每条向量文档包含 `owner_user_id`（字符串过滤字段）、`conversation_id`、`block_id`、主题与时间元数据及向量字段。索引定义中的向量字段名、维度和相似度类型必须与实际 Embedding 模型一致；在模型尚未选定前维度未知，不能给出或创建可信的维度配置。查询必须使用当前会话的 owner 过滤，再用返回的 ID 从 `LingGuangChat` 取回本人原文。

切换前还需实现并验证：语音框新增/编辑/删除时同步向量写入与清理，历史记录更新后的重嵌入，检索失败时回退到关键词搜索，以及跨用户过滤测试。MongoDB Atlas Vector Search 可存放向量索引，无需 Pinecone；本项目当前没有调用 Pinecone。
