load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "kernelized",
    srcs = ["kernelized.py"],
    srcs_version = "PY3",
    deps = []
)

py_library(
    name = "augmented_dense",
    srcs = ["augmented_dense.py"],
    srcs_version = "PY3",
    deps = []
)


py_library(
    name = "lstsq_models",
    srcs = ["lstsq_models.py"],
    srcs_version = "PY3",
    deps = [
        # "//utils:utils_impl",
        "//optimization/shared:projector_utils"
        ]
)

py_library(
    name = "nuclear_models",
    srcs = ["nuclear_models.py"],
    srcs_version = "PY3",
    deps = [
        # "//utils:utils_impl",
        "//optimization/shared:projector_utils"
        ]
)

py_library(
    name = "emnist_models",
    srcs = ["emnist_models.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:utils_impl",
        "//optimization/shared:projector_utils",
        "//utils/models:kernelized",
        "//utils/models:augmented_dense"
        ]
)

py_library(
    name = "bin_lr_models",
    srcs = ["bin_lr_models.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:utils_impl",
        "//optimization/shared:projector_utils",
        "//utils/models:augmented_dense"
        ]
)


