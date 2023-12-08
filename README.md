# NeRF-Flow: A High Performance Neural Rendering Framework

## Motivation
Neural radiance fields (NeRF) have emerged as a transformative advance in photorealistic view synthesis and novel view generation. It can be seen as the next generation of graphic render technique after ray-tracing and rasterization.

While recent methods have made progress, major challenges remain in neural rendering. Current NeRF implementations offer high visual quality but lack performance for fast rendering. Hardware accelerators can help, but have limited portability across platforms and lack adaptability across scenes. Algorithmic optimizations also accelerate NeRF queries, but reduce programmability and are not easily portable.

We see opportunities in a full stack end-to-end neural rendering framework that can match algorithmic and hardware gaps.

## Project Vision

We aims to make photo realistic rendering portable to all device and scalable to a wide range of environment. For example, VR devices and edge devices.

## Projects

- A NeRF Rendering Framework on GPU
- A Virtualizable Unified NeRF Spatial Processor
- A NeRF Compiler for NeRF Spatial Processor
- A Photo-Realistic Framwork on VR/AR Glasses

## Platforms

NeRF-Accelerator, GPU, TPU, CPU

## Organizers

- Jiongxing Shen (Interpreter Manager)
- Haimeng Ren (Processor Manager)
- Kaiyan Chang (Compiler Manager)
This project is a collaboration of dedicated professionals. Each core developer and contributor plays a unique role in the project. For any inquiries or questions, feel free to reach out to the corresponding developer based on their expertise. 
| Developer | Organization | Responsibility | Contact |
|-----------|--------------|----------------|---------|
| Jiongxing Shen | ZheJiang Lab | overall interpreter implementation |  |
| Haimeng Ren | Shanghai Tech University | NeRF processor, mapper implementation, auto scheduler | |
| Kaiyan Chang | Institute of Computing Technology, Chinese Academic of Science | compiler design | changkaiyan@live.com |

## How to use

You can use any instant-ngp model and datasets to test.

```shell
mkdir build
cd build
cmake ..
make -j8
cd ..
./nerftest ./xxx.ingp ./xxx.json
```

## Files

The main file is src/main.cu.

## Result

## Compare



## Acknowledgement

Welcome to contribute.



