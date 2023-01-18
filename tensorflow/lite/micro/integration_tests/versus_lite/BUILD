load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "conv_dilation_test",
    size = "small",
    srcs = [
        "conv_dilation_test.py",
    ],
    deps = [
        requirement("numpy"),
        requirement("pytest"),
        requirement("tensorflow-cpu"),
        "//python/tflite_micro:versus_lite",
    ],
)

py_test(
    name = "space_to_batch_test",
    size = "small",
    srcs = [
        "space_to_batch_test.py",
    ],
    deps = [
        requirement("numpy"),
        requirement("pytest"),
        requirement("tensorflow-cpu"),
        "//python/tflite_micro:versus_lite",
    ],
)
