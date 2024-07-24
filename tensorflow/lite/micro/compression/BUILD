load(
    "@flatbuffers//:build_defs.bzl",
    "flatbuffer_cc_library",
    "flatbuffer_py_library",
)
load("@rules_python//python:defs.bzl", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

flatbuffer_cc_library(
    # Generates the header-only library "metadata_generated.h", used to read
    # the metadata flatbuffer.
    name = "metadata_cc",
    srcs = ["metadata.fbs"],
)

flatbuffer_py_library(
    # Generates the Python module "metadata_py_generated", used to read the
    # metadata flatbuffer.
    name = "metadata_py",
    srcs = ["metadata.fbs"],
)

cc_test(
    name = "metadata_test_cc",
    size = "small",
    srcs = ["metadata_test.cc"],
    deps = [
        ":metadata_cc",
        "//tensorflow/lite/micro:hexdump",
        "//tensorflow/lite/micro/testing:micro_test",
        "@flatbuffers//:runtime_cc",
    ],
)

py_test(
    name = "metadata_test_py",
    size = "small",
    srcs = ["metadata_test.py"],
    main = "metadata_test.py",
    deps = [
        "metadata_py",
        "@flatbuffers//:runtime_py",
        requirement("hexdump"),
    ],
)
