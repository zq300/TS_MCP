# TS_MCP Server
使用Chronos时序预测大模型对各种时序数据进行预测
## 运行
```
pip install -r requirements.txt
```
uv/conda 均可

### 单次执行
```
fastmcp run server.py --transport streamable-http --port 9000 --host 0.0.0.0
```

### 后端运行
#### nohup
```
nohup fastmcp run server.py --transport streamable-http --port 9000 --host 0.0.0.0 > /raid/users/zq/ts_mcp/mcp_stdout.log 2>&1 &
```

#### tmux 或 screen
```
tmux new -s fastmcp
```
```
screen -S fastmcp
```

##### 使用Cline的配置文件
```json
{
  "mcpServers": {
    "chronos": {
      "disabled": false,
      "timeout": 60,
      "transportType": "stdio",
      "command": "conda",
      "args": [
        "run",
        "-n",
        "chronos_mcp",
        "--no-capture-output",
        "python",
        "./server.py",
        "--transport",
        "stdio"
      ]
    }
  }
}

```

#### 使用cherry studio的配置文件
1、可流式传输的HTTP
2、部署后的URL链接：http://域名:9000/mcp