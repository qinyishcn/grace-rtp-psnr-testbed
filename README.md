# GRACE RTP / AITestbed PSNR simulation

基于 [5G-MAG/6G-Testbed](https://github.com/5G-MAG/6G-Testbed) 的原版 `netemu`、`TrafficLogger` 和 aiortc，研究 GRACE 视频特征按块受损与帧级 PSNR 的关系。

已完成真实 GRACE 权重运行、十块独立压缩与 RTP 分片、截止时间恢复、真实 ICE/DTLS/SRTP/netem 预检、12,096 条 P 帧仿真标签及按视频留出的预测对照。I 帧用传统 JPEG 保证交付，P 帧无 I-patch。新增代码独立维护，不修改既有 Testbed checkout。

- [实验结果与限制](results/report_zh.md)
- [完整思路、RTP 重传、jitter buffer 与在线适配分析](docs/design_zh.md)
- [逐帧数据](results/frame_dataset.csv)、[预测指标](results/prediction/prediction_metrics.csv)、[真实 SRTP 包头](results/actual_srtp_packet_headers.csv)
- [网络参数敏感性图](results/network_sensitivity.png)

这是 256×192、12 个短视频片段的探索性原型。大批数据来自序列化 RTP 字节上的虚拟 FIFO 仿真；原版 Testbed 的真实 netem 测试单独报告，不能用虚拟样本数量代替真实抓包规模。未验证公网、长期分布漂移、在线编解码吞吐量或 TTA 收益。

## 环境与外部依赖

新建 Python 环境安装 `pip install -e .`。使用 CUDA 版本 PyTorch 和匹配 torchvision；已验证环境见 [runtime_sources.json](results/runtime_sources.json)。另需 `imageio pypng torchac ninja requests beautifulsoup4`，以及 Linux 的 `iproute2 tcpdump sudo`。Windows 可运行预测/报告，真实 Testbed netem 使用 Linux/WSL。

GRACE 代码与权重受其独立许可约束，不随本仓库分发；请先确认[上游许可](https://github.com/UChi-JCL/Grace/blob/14f1721663acd5d16ca776ce61b4a5860167d24d/LICENSE)适用于自己的使用情形，然后自行获取：

```bash
git clone https://github.com/UChi-JCL/Grace assets/Grace
git -C assets/Grace checkout 14f1721663acd5d16ca776ce61b4a5860167d24d
python scripts/fetch_selected_models.py --output-root assets/Grace/models
git clone https://github.com/5G-MAG/6G-Testbed assets/6G-Testbed
git -C assets/6G-Testbed checkout 8581bf4974ccb35a6cc14791caebe7fcae4a0502
```

权重下载器读取官方公开 Google Drive tar 的 Range 元数据，下载两个 checkpoint 和光流初始化数组；每次 Range 检查返回范围和长度，最多三次网络重试。最终使用的 4096 模型 SHA-256 为 `cb10ab522ed1bcd74562d04bcffc1283d602cd585ceaf9708204bb5ddb041baa`。模型初始化后逐参数核对实际加载权重，禁止静默缺键形成随机模型。

本次素材为已有的 [StreamingBench](https://github.com/THUNLP-MT/StreamingBench) Real-Time Visual Understanding 视频 1–12，文件命名 `001.mp4` 至 `012.mp4`，每个取第 10 秒开始的 16 帧，按 30 fps 就近选原生帧并缩放。素材需自行获取；[帧/特征哈希](results/source_feature_hashes.csv)供确认选样，元数据见 `results/source_manifest_q*.json`。视频、原始帧和模型不上传。

## 分开运行发送与接收阶段

```bash
python -m grace_rtp.experiment prepare \
  --grace assets/Grace --checkpoint assets/Grace/models/grace/4096_freeze.model \
  --videos-dir /your/videos --videos 12 --frames 16 --q 1 --scale 1 --cache cache/q1
python -m grace_rtp.experiment run \
  --grace assets/Grace --checkpoint assets/Grace/models/grace/4096_freeze.model \
  --cache cache/q1 --seeds 3 --trace --output results/pilot_q1
```

`prepare` 只负责源侧编码与干净发送参考；`run` 以收到的独立块构建接收参考，原图仅用于离线评分。独立块使用 DEFLATE，可逆还原真实 GRACE 符号；不声称等于原论文学习熵编码的码率。`--fixed-loss 0.2 --profiles configs/controlled.yaml` 用于同 GOP 固定掩码受控标定，与真实逐包扰动数据分开。

完整虚拟试验可设置以下环境后运行 `bash scripts/run_pilot.sh`。脚本复用存在的 cache，换模型/素材/参数时请使用新的 cache 路径，避免混用。

```bash
export GRACE_PYTHON=/path/to/venv/bin/python
export GRACE_ROOT=/absolute/path/to/Grace
export GRACE_VIDEOS=/absolute/path/to/videos
bash scripts/run_pilot.sh
python -m grace_rtp.predict results/pilot_q1/frames.csv results/pilot_q2/frames.csv
python scripts/summarize.py
```

训练会在本机产生 `results/prediction/model.joblib`，可用 `grace_rtp.infer.Predictor` 做有状态逐帧推理，或运行 `python -m grace_rtp.infer --model ... --input features.jsonl --output estimates.jsonl`。输入只需上述捕获特征、协议配置、`gop_id` 与 `stream_id`，按流内帧顺序提交；新 GOP 自动重置历史。模型文件不上传，公开 CSV 可在几秒内重新训练。joblib 文件只加载自己生成的可信模型，并保持 scikit-learn 版本一致。

## 原版 AITestbed 真实网络通路

先安装 Testbed 的 `netemu` 和 aiortc fork；新建环境时避免与已运行项目共享依赖。真实接入采用 `grace_rtp.testbed.GraceScenario`、原 `TrafficLogger` 及原 `NetworkEmulator`，无需 LLM 服务。它是独立扩展 runner，没有自动改写上游 orchestrator 的 scenario registry。

```bash
export TESTBED_ROOT=/absolute/path/to/6G-Testbed
# 还需设置 GRACE_PYTHON、GRACE_ROOT；cache/q1 应已准备好。
sudo -E bash scripts/run_isolated_netem.sh
python scripts/audit_pcap.py
```

脚本创建独立 network namespace，仅在其中的 lo 上应用四种原始 profile，I 帧和完整布局通过可靠 DataChannel 在扰动前交付。自定义收端不发 NACK/RTX，按可靠布局指定的帧截止时间接收完整块；神经解码随后离线进行。namespace 内 ICE 主机候选限制为 127.0.0.1，不触及宿主网卡。退出清理命名空间及抓包进程。

原始 PCAP、SQLite 留在本机；公开 RTP 基本包头、tc 状态、完整块掩码及特征哈希。`rtc.py` 也提供无 kernel netem 的真实 SRTP/虚拟调度诊断入口。使用 aiortc 私有 DTLS 接口，升级版本必须重新测试。

## 验证与预测边界

```bash
python -m unittest discover -s tests -v
python -m scripts.benchmark --grace assets/Grace --checkpoint assets/Grace/models/grace/4096_freeze.model
```

Testbed fork 下本地 15 项检查通过；普通 aiortc 下四项 fork 行为检查会明确跳过，其余传输与真实 SRTP 检查仍运行。预测仅用帧截止前可见包；`final_capture_loss`、真实掩码、PSNR 和 profile 名称均不作为网络输入。配对抓包模型需要同步的收发观测和已知协议时限/GOP/质量配置；SI/TI 与 latent 统计增强模型需要显式源侧反馈。纯上游抓包模型另作基线。权重哈希、源身份、逐帧缺块与不理想结果均保留。

The Software used in this research was created by Yihua Cheng of UChicago. © 2023 UChicago. Original implementation files in this repository use MIT; external dependencies retain their respective licenses.
