cmake_minimum_required(VERSION 3.7...3.13)

neurocf_add_example(stress_cpu High_API/CPU/stress.cpp)
neurocf_add_example(stress_gpu High_API/GPU/stress.cpp)
neurocf_add_example(simple_cpu High_API/CPU/simple.cpp)
neurocf_add_example(simple_gpu High_API/GPU/simple.cpp)
neurocf_add_example(external_layers_cpu High_API/CPU/external_layers.cpp)
neurocf_add_example(external_layers_gpu High_API/GPU/external_layers.cpp)