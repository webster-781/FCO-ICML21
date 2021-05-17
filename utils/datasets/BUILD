load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "emnist_dataset",
    srcs = ["emnist_dataset.py"],
    srcs_version = "PY3",
)

py_test(
    name = "emnist_dataset_test",
    srcs = ["emnist_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [":emnist_dataset"],
)


py_library(
    name = "synthetic_dataset",
    srcs = ["synthetic_dataset.py"],
    srcs_version = "PY3",
    deps = [],
)