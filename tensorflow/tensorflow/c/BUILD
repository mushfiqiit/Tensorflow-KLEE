load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

cc_library(
    name = "tf_tensor_lib",
    srcs = [
        "tf_tensor.cc",
    ],
    hdrs = [
        "tf_tensor.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "tf_tensor_harness",
    srcs = ["tf_tensor_harness.cc"],
    deps = [
        ":tf_tensor_lib",
    ],
    copts = ["-Iexternal/klee/include"],
    linkopts = ["-L$(HOME)/tools/klee/build/lib", "-lkleeRuntest"],  # You might need to adjust this
    visibility = ["//visibility:public"],
)
