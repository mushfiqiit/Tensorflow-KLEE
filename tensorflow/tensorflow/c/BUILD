load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

cc_library(
    name = "tf_tensor_lib",
    srcs = [
        "tf_tensor.cc",
    ],
    hdrs = [
        "tf_tensor.h",
        "tf_datatype.h",
        "tf_status.h",
        "tf_status_helper.h",
        "c_api_macros.h",
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
        "@klee//:klee",
        "//tensorflow/core/platform:platform_port",
        "//third_party/xla/xla/tsl/c:tsl_status_headers",
    ],
    linkopts = ["-L/home/mushfiqur/tools/klee/build/lib", "-lkleeRuntest"],
    visibility = ["//visibility:public"],
)