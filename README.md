# Experiment-based Roofline Drawer 📊

This tool provides a comprehensive way to visualize the performance limits of your hardware.
It supports both **Spec-based** theoretical rooflines and **Experiment-based** empirical rooflines.

## Features
- Draw Roofline models using hardware specifications (TFLOPS, GB/s).
- Measure and plot actual hardware performance using micro-benchmarks.
- Compare your custom kernels against the hardware's peak.

- Cache size estimation: Experimentally measure L1, L2, L3 size
- Stride access 실험 - 결과 propfiler에 넣으면 어디서 bank conflict나는지 등 확인할 수 있도록 


## Working Docs
- **[ Roofline Model Basics ]**: https://docs.google.com/document/d/1uz4m0aX36uUriByhQ4vWar1nk-JponozvvGUXU_GdpE/edit?tab=t.0

- **[ HW Profiling Tool Collection ]**: https://docs.google.com/document/d/1lEWg12gvInNzhY4uPANddT7mrgAkPFMHj6JVoEY8BeE/edit?tab=t.0