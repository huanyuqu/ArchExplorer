# GPU 组件与本地资源

GPU 的 SM、L1/shared、L2、互连和 DRAM 参数由原生 Accel-Sim 配置编译器消费。
接口与字段来源见[原生引擎文档](accelsim.md)，完整程序用法见[ViT/CLIP demo](tutorials/05-full-models.md)。

自建 GPU 本地带宽/吞吐时延路径已移除。旧 `gpu_sm` 配置不迁移成 vector 单元继续执行，
也没有旧 GPU 预测 fallback。非 GPU vector、systolic、PIM 的通用 service 语义仍服务各自后端。
