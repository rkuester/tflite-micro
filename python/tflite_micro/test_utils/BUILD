load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "crosscheck",
    srcs = [
        "crosscheck.py",
    ],
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
    ],
)

py_test(
    name = "crosscheck_test",
    size = "small",
    srcs = [
        "crosscheck_test.py",
    ],
    data = [
        "//tensorflow/lite/micro/examples/hello_world:hello_world.tflite",
    ],
    deps = [
        ":crosscheck",
    ],
)
