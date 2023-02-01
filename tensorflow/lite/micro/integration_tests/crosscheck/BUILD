load("@tflm_pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "separable_conv_2d_test",
    size = "small",
    srcs = [
        "separable_conv_2d_test.py",
    ],
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//python/tflite_micro/test_utils:crosscheck",
    ],
)
