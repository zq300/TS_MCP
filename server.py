
from fastmcp import FastMCP
import shutil
import os
import torch
import pandas as pd
import logging
from chronos import BaseChronosPipeline

# ===== 日志配置 =====
LOG_PATH = "/raid/users/zq/ts_mcp/server.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logging.info("==== MCP Server Starting ====")

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

logging.info(f"Loading Chronos model: {MODEL_PATH} on {DEVICE} ...")
pipeline = BaseChronosPipeline.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,
    torch_dtype=DTYPE,
)
logging.info("Model loaded.")

DATA_DIR = "/raid/users/zq/ts_mcp/data"

# ===== 文件上传工具 =====
@mcp.tool(description="上传CSV文件到服务器指定目录，已有文件不覆盖")
def upload_csv_file(
    file_name: str,
    content: str,
) -> dict:
    save_path = os.path.join(DATA_DIR, file_name)
    if os.path.isfile(save_path):
        logging.warning(f"文件已存在，不保存新内容：{save_path}")
        return {"success": False, "msg": f"文件 {file_name} 已存在，未保存新上传内容。"}
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"已写入文件：{save_path}")
        return {"success": True, "msg": f"文件已保存到 {save_path}"}
    except Exception as e:
        logging.error(f"保存失败: {e}")
        return {"success": False, "msg": f"保存失败: {e}"}


# ===== 文件查看 ====
@mcp.tool(description="查看数据目录下的所有文件")
def list_uploaded_files() -> dict:
    files = os.listdir(DATA_DIR)
    logging.info(f"列出数据目录所有文件：{files}")
    return {"files": files}

# ===== 预测工具 =====
@mcp.tool(description="从csv文件读取序列，预测未来N步的分位点与均值")
def predict_quantiles_from_csv(
    file_name: str,
    col_name: str = "#Passengers",
    prediction_length: int = 12,
    quantile_levels: list[float] = (0.1, 0.5, 0.9),
) -> dict:
    csv_path = os.path.join(DATA_DIR, file_name)
    logging.info(f"拼接后的csv路径: {csv_path}")

    if not os.path.isfile(csv_path):
        logging.error(f"File not found: {csv_path}")
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if col_name not in df.columns:
        logging.error(f"列 {col_name} 不在csv中，实际可选列: {df.columns.tolist()}")
        raise ValueError(f"列 {col_name} 不在csv中，实际可选列: {df.columns.tolist()}")
    series = df[col_name]
    tensor = torch.tensor(series.values, dtype=torch.float32)
    quantiles, mean = pipeline.predict_quantiles(
        context=tensor,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    logging.info(f"预测完成: file={file_name}, col={col_name}, steps={prediction_length}, quantiles={quantile_levels}")
    return {
        "quantiles": quantiles.tolist(),
        "mean": mean.tolist(),
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
