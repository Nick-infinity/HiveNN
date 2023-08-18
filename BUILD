cc_library(
    name = "hnn",
    srcs = [
        "src/common/helper.h",
        "src/common/runcontext.cpp",
        "src/kernels/cpu/fully_connected.cpp",
        "src/kernels/cpu/op_utils.h",
        "src/kernels/cpu/rms_normalization.cpp",
        "src/kernels/cpu/soft_max.cpp",
        "src/kernels/cpu/dequant_token.cpp",
        "src/memory_manager/buffer_allocator.cpp",
        "src/memory_manager/buffer_manager.cpp",
        "src/memory_manager/tensor.h",
        "src/model_structure/configs/configs.h",
        "src/model_structure/configs/transformers_config.h",
        "src/model_structure/models/model.h",
        "src/model_structure/models/transformers/llama_2.cpp",
        "src/parser/model_parser.cpp",
        "src/types.h",
        "src/utils.cpp",
         "src/kernels/cpu/argmax.cpp",
    ],
    hdrs = [
        "src/common/runcontext.h",
        "src/kernels/cpu/fully_connected.h",
        "src/kernels/cpu/operators.h",
        "src/kernels/cpu/rms_normalization.h",
        "src/kernels/cpu/soft_max.h",
        "src/kernels/cpu/dequant_token.h",
        "src/memory_manager/buffer_allocator.h",
        "src/memory_manager/buffer_manager.h",
        "src/model_structure/models/model.h",
        "src/model_structure/models/transformers/llama_2.h",
        "src/parser/model_parser.h",
        "src/utils.h",
        "src/kernels/cpu/argmax.h",
    ],
    copts = [
        "-Isrc/",
        "-Ofast",
        "-fopenmp",
        "-static-openmp",
    ],
    linkopts = [
        "-Wl,-soname,libhnn.so",
        "-Wl,-Bstatic -lomp -Wl,-Bdynamic",
    ],
)

cc_binary(
    name = "Hnn_Test",
    srcs = [
        "hnn_test/hnn_test.cpp",
    ],
    copts = [
        "-Isrc/",
        "-Ofast",
        "-fopenmp",
        "-static-openmp",
    ],
    linkopts = ["-Wl,-Bstatic -lomp -Wl,-Bdynamic"],
    linkstatic = 1,
    deps = [":hnn"],
)

cc_binary(
    name = "quantizer",
    srcs = ["quantizer.c"],
)
