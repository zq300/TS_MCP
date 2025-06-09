from fastmcp import FastMCP
import os
import torch
import pandas as pd
from chronos import BaseChronosPipeline

# ===== MCP 实例 =====
mcp = FastMCP(
    name="Chronos Forecaster",
    description="Zero-shot probabilistic forecasting via Amazon Chronos",
    dependencies=[
        "torch==2.2.0",
        "pandas",
        "safetensors>=0.4.3",
        "transformers>=4.40,<4.41",
        "accelerate>=0.26",
        "chronos-forecasting @ git+https://github.com/amazon-science/chronos-forecasting.git",
        "flash-attn==0"
    ],
)

# ====== 只加载一次模型 ======
MODEL_PATH = os.getenv("CHRONOS_MODEL", "/raid/users/zq/ts_mcp/chronos_t5_small")
DEVICE = os.getenv("CHRONOS_DEVICE", "cpu")  # 可设为"cuda"
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32

print(f"Loading Chronos model: {MODEL_PATH} on {DEVICE} ...")
pipeline = BaseChronosPipeline.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,
    torch_dtype=DTYPE,
)
print("Model loaded.")

# ===== 工具实现 =====
@mcp.tool(description="从csv文件读取序列，预测未来N步的分位点与均值")
def predict_quantiles_from_csv(
    csv_path: str,
    col_name: str = "#Passengers",
    prediction_length: int = 12,
    quantile_levels: list[float] = (0.1, 0.5, 0.9),
) -> dict:
    """
    Parameters
    ----------
    csv_path : str     输入csv文件路径
    col_name : str     要分析的列名，默认为"#Passengers"
    prediction_length : int 预测步数
    quantile_levels : list[float] 需要返回的分位点
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if col_name not in df.columns:
        raise ValueError(f"列 {col_name} 不在csv中，实际可选列: {df.columns.tolist()}")
    series = df[col_name]
    tensor = torch.tensor(series.values, dtype=torch.float32)
    quantiles, mean = pipeline.predict_quantiles(
        context=tensor,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    return {
        "quantiles": quantiles.tolist(),
        "mean": mean.tolist(),
    }

# ===== 入口 =====
if __name__ == "__main__":
    mcp.run(transport="stdio")
