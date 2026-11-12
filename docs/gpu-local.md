# GPU Components and Local Resources

GPU SM, L1/shared, L2, interconnect, and DRAM parameters are consumed by the native Accel-Sim configuration compiler. For interfaces and field sources, see [native engine documentation](accelsim.md); for complete program usage, see [ViT/CLIP demo](tutorials/05-full-models.md).

The self-built GPU local bandwidth/throughput latency path has been removed. Old `gpu_sm` configurations are not migrated to vector units for continued execution, and there is no old GPU prediction fallback. Non-GPU vector, systolic, and PIM common service semantics still serve their respective backends.
