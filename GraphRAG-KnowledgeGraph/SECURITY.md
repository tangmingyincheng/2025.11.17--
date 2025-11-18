# 安全配置说明

## ⚠️ 重要提示

本项目包含需要配置的敏感信息，**切勿直接提交到公开仓库**。

## 🔐 配置步骤

### 1. 配置环境变量

复制 `.env.example` 为 `.env` 并填入真实值：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# OpenAI API 配置
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# Neo4j 配置
NEO4J_PASSWORD=your_secure_password_here
```

### 2. 配置 config.yaml

编辑 `configs/config.yaml`：

```yaml
llm:
  api_key: YOUR_API_KEY_HERE  # 替换为真实 API Key
```

### 3. 配置 Docker Compose

如果使用自定义密码，在启动前设置环境变量：

```bash
export NEO4J_PASSWORD=your_password
docker-compose up -d
```

或直接在 `.env` 文件中配置（已包含在 `.gitignore` 中）。

## 📋 受保护的文件

以下文件已添加到 `.gitignore`，不会被提交：

- `.env` - 环境变量配置
- `*.log` - 日志文件
- `neo4j_data/` - Neo4j 数据目录
- `__pycache__/` - Python 缓存

## ✅ Git 检查清单

提交代码前，确保：

- [ ] `.env` 文件未被追踪
- [ ] `config.yaml` 中无真实 API Key
- [ ] 代码中无硬编码密码
- [ ] `.gitignore` 正确配置

## 🔍 验证命令

检查是否有敏感信息泄露：

```bash
# 检查暂存区
git diff --cached

# 搜索 API Key
git grep -i "sk-"

# 搜索密码
git grep -i "password123"
```

## 📖 参考

- [GitHub 安全最佳实践](https://docs.github.com/en/code-security)
- [环境变量管理](https://12factor.net/config)
