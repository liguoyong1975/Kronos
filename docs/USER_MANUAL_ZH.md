# Kronos 使用说明书

## 1. 项目概览
Kronos 是一套面向金融 K 线数据的基础模型体系，提供从原始行情数据离散化、序列建模、到预测结果可视化的一站式工具。仓库内同时包含模型推理、数据预处理、微调、回测以及 Web 端演示等模块，适合量化研究员、数据科学家和工程团队快速构建行情预测或策略验证原型。

## 2. 系统环境要求
- **操作系统**：Windows 10/11、macOS 12+ 或任意主流 Linux 发行版。
- **Python**：3.10 及以上版本。
- **硬件建议**：
  - 推理：≥8GB 内存；可选 GPU（CUDA 11+）。
  - 训练/微调：>=12GB 显存的 NVIDIA GPU，或使用多 GPU 集群。
- **依赖安装**：
  ```bash
  pip install -r requirements.txt
  ```
  若需运行 Web UI，请额外安装 `webui/requirements.txt` 中的依赖。

## 3. 仓库目录速览
| 路径 | 说明 |
| --- | --- |
| `model/` | 核心模型、分词器与预测器实现。 |
| `finetune/` | 面向 Qlib 数据的端到端微调、回测脚本。 |
| `finetune_csv/` | 基于 CSV 数据的轻量级微调示例与配置。 |
| `examples/` | notebook 与脚本示例，演示模型用法。 |
| `webui/` | Streamlit Web 演示，包括实时预测展示。 |
| `tests/` | 单元测试与集成测试集合。 |
| `figures/` | 文档插图、流程图等资源。 |

## 4. 快速上手
### 4.1 下载预训练模型
- 在 Hugging Face Hub 上选择合适的模型与分词器，例如 `NeoQuasar/Kronos-small` 与 `NeoQuasar/Kronos-Tokenizer-base`。
- 将模型缓存到本地，或在代码中直接使用 `.from_pretrained()` 拉取。

### 4.2 最小可运行示例
```python
from model import KronosTokenizer, Kronos, KronosPredictor
import pandas as pd

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 准备数据
raw_df = pd.read_csv("./data/XSHG_5min_600977.csv")
raw_df["timestamps"] = pd.to_datetime(raw_df["timestamps"])

lookback, pred_len = 400, 120
x_df = raw_df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = raw_df.loc[:lookback-1, 'timestamps']
y_timestamp = raw_df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 推理
forecast = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1
)
print(forecast.head())
```

## 5. 核心组件说明
- **KronosTokenizer**：混合 Transformer + Binary Spherical Quantization 的离散化模块，负责将多维连续行情数据编码为层级离散 token，并支持反量化重建。【F:model/kronos.py†L13-L131】
- **Kronos**：自回归 Transformer 模型，接收量化后的 token 序列进行建模，并支持可学习时间嵌入和多种 dropout 策略。【F:model/kronos.py†L134-L344】
- **KronosPredictor**：封装数据预处理、归一化、推理及反归一化流程，为研发人员提供统一的预测接口。【F:model/kronos.py†L347-L596】
- **model_dict / get_model_class**：集中式模型注册表，便于通过字符串名称反射加载模型组件。【F:model/__init__.py†L1-L16】

## 6. 数据准备
1. **K 线结构**：默认使用包含 `open`、`high`、`low`、`close`、`volume`、`amount` 字段的 DataFrame，时间戳需为 `DatetimeIndex` 或 `pandas.Series`。
2. **特征归一化**：预测器在内部自动完成归一化与反归一化，也可通过传入 `normalize_cfg` 自定义配置。
3. **时间特征**：若使用微调脚本，可通过配置文件自动生成 `minute/hour/weekday/day/month` 等时间特征。【F:finetune/config.py†L18-L48】
4. **数据裁剪**：注意模型的 `max_context` 限制，超过长度会自动截断；推荐在数据准备阶段控制窗口大小。

## 7. 微调流程
### 7.1 基于 Qlib 的流程
1. **配置编辑**：在 `finetune/config.py` 中修改数据路径、时间区间、批大小、学习率等参数。【F:finetune/config.py†L4-L109】
2. **数据预处理**：使用 `finetune/qlib_data_preprocess.py` 拉取 Qlib 数据并生成标准化样本。
3. **训练分词器**：运行 `python finetune/train_tokenizer.py --config finetune/config.py`，复用或更新预训练分词器权重。
4. **训练预测器**：运行 `python finetune/train_predictor.py --config finetune/config.py`，使用最新的 tokenizer 与预测器权重。
5. **验证与回测**：借助 `finetune/qlib_test.py` 和 `finetune/utils` 下的工具进行验证、指标计算与组合回测，结果会保存在 `outputs/` 中。
6. **在线推理**：训练完成后，将 `Config.finetuned_tokenizer_path` 与 `Config.finetuned_predictor_path` 替换到在线服务或脚本中。【F:finetune/config.py†L111-L147】

### 7.2 基于 CSV 的快速微调
- 当没有 Qlib 环境时，可使用 `finetune_csv/` 目录内的示例：
  1. 在 `finetune_csv/data/` 中放置样本 CSV；
  2. 通过 `finetune_csv/configs/*.yaml` 自定义窗口、特征与训练参数；
  3. 运行 `python finetune_csv/finetune_tokenizer.py` 或 `python finetune_csv/finetune_base_model.py` 启动训练；
  4. 可参考 `finetune_csv/examples/` 中的 notebook 做快速验证。

## 8. Web UI 演示
1. 安装前端依赖：`pip install -r webui/requirements.txt`。
2. 运行 `bash webui/start.sh` 或 `python webui/run.py` 启动 Streamlit 服务。【F:webui/start.sh†L1-L6】【F:webui/run.py†L1-L59】
3. 在浏览器中访问默认地址 `http://localhost:8501`，可查看预测曲线、置信区间和历史结果。`webui/templates/` 提供主题配置与可自定义组件。

## 9. 示例与测试
- `examples/` 中的 notebook 展示了特定市场、时间尺度下的完整推理流程，可作为二次开发的起点。
- 建议在改动核心逻辑后运行测试：
  ```bash
  pytest tests
  ```

## 10. 常见问题与排错
| 问题 | 解决方案 |
| --- | --- |
| 预测结果全为 NaN | 检查输入数据是否含缺失值；确认 `pred_len` 与 `y_timestamp` 长度一致。 |
| 上下文长度错误 | 调整 `lookback`，确保不超过模型 `max_context`；或裁剪输入序列。 |
| 微调无法收敛 | 降低学习率、增大 `n_train_iter`，或检查数据归一化范围。 |
| Web UI 启动失败 | 确认前端依赖已安装，并检查端口冲突。 |

## 11. 推荐工作流
1. 通过 `examples/` 或最小示例验证模型预测效果。
2. 使用 `finetune/` 或 `finetune_csv/` 在目标数据集上微调，并结合回测评估策略收益。
3. 将训练好的模型部署到 `webui/` 或企业内部服务，实现实时监控与交互式分析。

## 12. 许可证与引用
- 本项目采用 [Apache 2.0](../LICENSE) 许可证。
- 如在研究或产品中使用 Kronos，请在文献或产品说明中引用项目主页及官方论文。

## 13. 反馈与支持
- 问题反馈：通过 GitHub Issues 提交复现步骤与日志。
- 功能需求：欢迎提交 Pull Request 或在 Issue 中讨论。
- 社区交流：关注项目主页公布的社区渠道，或联系维护者参与共建。
