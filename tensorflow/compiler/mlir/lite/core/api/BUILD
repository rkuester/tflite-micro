load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/micro:build_def.bzl", "tflm_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "error_reporter",
    srcs = ["error_reporter.cc"],
    hdrs = ["error_reporter.h"],
    copts = tflite_copts(),
    deps = [],
)
