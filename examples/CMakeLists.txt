cmake_minimum_required(VERSION 3.7...3.13)

neurocf_add_example(simple_cpu High_API/simple_cpu.cpp)
neurocf_add_example(simple_gpu High_API/simple_gpu.cpp)
neurocf_add_example(hybrid_cpu High_API/hybrid_cpu.cpp)
neurocf_add_example(hybrid_gpu High_API/hybrid_gpu.cpp)