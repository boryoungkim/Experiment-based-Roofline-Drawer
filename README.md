# Experiment-based Roofline Drawer 📊

This tool provides a comprehensive way to visualize the performance limits of your hardware.
It supports both **Spec-based** theoretical rooflines and **Experiment-based** empirical rooflines.

## Features
- Draw Roofline models using hardware specifications (TFLOPS, GB/s).
- Measure and plot actual hardware performance using micro-benchmarks.
- Compare your custom kernels against the hardware's peak.

- (Only CPU, NPU- TODO)Cache size estimation: Experimentally measure L1, L2, L3 size
- (TODO) Stride access - bank conflict experiment


## Working Docs
- **[ Roofline Model Basics ]**: https://docs.google.com/document/d/1uz4m0aX36uUriByhQ4vWar1nk-JponozvvGUXU_GdpE/edit?tab=t.0
- **[ Memory Basics ]**: https://docs.google.com/document/d/1lEWg12gvInNzhY4uPANddT7mrgAkPFMHj6JVoEY8BeE/edit?tab=t.0
- **[ Roofline Analysis ]**:https://docs.google.com/document/d/1SDAaJo_vAAjAan_cM-NrdSsi1yPg9HCMAZdtxEnc9B0/edit?tab=t.0
