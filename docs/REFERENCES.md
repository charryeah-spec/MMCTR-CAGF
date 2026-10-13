# 参考与引用

本项目在以下开源工作与赛题基线之上构建，使用均已保留原始协议与署名。

## 赛题与数据

1. **WWW2025 MM-CTR Challenge**  
   https://erel-mir.github.io/challenge/mmctr-track2/

   官方任务定义：MicroLens-1M（约 100 万用户、91K 物品）、128 维 Task 1 输出、Task 2 单神经模型约束、Task 1&2 分阶段训练与 AUC 评估。

2. **MicroLens-1M 数据集**  
   Ni Y, Cheng Y, Liu X, et al. *A Content-Driven Micro-Video Recommendation Dataset at Scale.* arXiv:2305.14099, 2023.

## 框架与基线代码

3. **FuxiCTR** — CTR 训练框架  
   Zhu J, Liu J, Yang S, et al. *Open Benchmarking for Click-Through Rate Prediction.* CIKM, 2021.  
   https://github.com/reczoo/FuxiCTR

4. **WWW2025_MMCTR_Challenge 官方 Baseline** — DIN 模型、数据加载、训练入口  
   https://github.com/reczoo/WWW2025_MMCTR_Challenge

## 参考实现的模型结构

5. **Transformer_DCN** — Task 2 CTR 架构参考（Apache 2.0）  
来源：WWW 2025 MM-CTR Task 2 公开第一名方案  
   https://github.com/pinskyrobin/WWW2025_MMCTR  
   Xu J, Zhao Z, Hu X, Song Z. *1st Place Solution of WWW 2025 EReL@MIR Workshop Multimodal CTR Prediction Challenge.* arXiv:2505.03543, 2025.  
   https://arxiv.org/abs/2505.03543  
   本地文件：`src/Transformer_DCN.py`

6. **RQ 残差量化思路** — Task 1 嵌入增强参考  
   同上开源仓库中的量化重建思路；本地为独立离线实现：`src/embedding/residual_quantizer.py`

## 本项目原创部分

以下模块为本项目设计与实现，未直接拷贝上游代码：

- `src/embedding/build_behavior_emb.py` — 行为共现图 + SVD
- `src/embedding/build_hybrid_emb.py` — 四源加权融合
- `src/embedding/fuse_item_info.py` — 双路嵌入融合工具
- `scripts/check_compliance.py` — 赛题合规自检
- `scripts/build_hybrid_embedding.sh` 及 `run_task*.sh` — 实验流水线
- `docs/`、`report/` — 文档与实验报告

## 相关论文

- Zhou G, et al. *Deep Interest Network for Click-Through Rate Prediction.* KDD, 2018.（DIN）
- Wang R, et al. *DCN V2: Improved Deep & Cross Network.* WWW, 2021.（DCN v2）
