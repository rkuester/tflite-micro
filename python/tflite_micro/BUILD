load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "versus_lite",
    srcs = [
        "versus_lite.py",
    ],
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
    ],
)

py_binary(
    name = "versus_lite.crosscheck",
    srcs = [
        "versus_lite.py",
    ],
    deps = [
        ":versus_lite",
    ],
    main = "versus_lite.py",
)
