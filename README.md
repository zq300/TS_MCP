# TS_MCP Server
使用Chronos时序预测大模型对各种时序数据进行预测
### Configure MCP Server
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
        "/raid/users/zq/ts_mcp/server.py",
        "--transport",
        "stdio"
      ]
    }
  }
}
