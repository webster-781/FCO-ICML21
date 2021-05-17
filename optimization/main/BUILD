load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "centralized_trainer",
    srcs = ["centralized_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//optimization/emnist:centralized_emnist",
        "//optimization/bin_lr:centralized_bin_lr",
        "//optimization/shared:optimizer_utils",
        "//utils:utils_impl",
        "//utils:threading_utils"
    ],
)


py_binary(
    name = "federated_trainer",
    srcs = ["federated_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//optimization/emnist:federated_emnist",
        "//optimization/shared:fed_avg_schedule",
        "//optimization/shared:fed_dual_avg_schedule",
        "//optimization/shared:optimizer_utils",
        "//optimization/lstsq:federated_lstsq",
        "//optimization/bin_lr:federated_bin_lr",
        "//optimization/nuclear:federated_nuclear",
        "//utils:utils_impl",
        "//utils:threading_utils"
    ],
)


