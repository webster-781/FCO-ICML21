load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "federated_nuclear",
    srcs = ["federated_nuclear.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:training_loop",
        "//utils:training_utils",
        "//utils/datasets:synthetic_dataset",
        "//utils/models:nuclear_models",
    ],
)