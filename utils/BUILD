load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "aggregate_fns",
    srcs = ["aggregate_fns.py"],
    srcs_version = "PY3",
)

py_test(
    name = "aggregate_fns_test",
    srcs = ["aggregate_fns_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":aggregate_fns"],
)

py_library(
    name = "centralized_training_loop",
    srcs = ["centralized_training_loop.py"],
    srcs_version = "PY3",
    deps = [
        ":utils_impl",
        "//optimization/shared:keras_callbacks",
    ],
)

py_library(
    name = "checkpoint_manager",
    srcs = ["checkpoint_manager.py"],
    srcs_version = "PY3",
)

py_test(
    name = "checkpoint_manager_test",
    srcs = ["checkpoint_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":checkpoint_manager"],
)

py_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "checkpoint_utils_test",
    size = "large",
    srcs = ["checkpoint_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":checkpoint_utils"],
)

py_library(
    name = "metrics_manager",
    srcs = ["metrics_manager.py"],
    srcs_version = "PY3",
    deps = [":utils_impl"],
)

py_test(
    name = "metrics_manager_test",
    srcs = ["metrics_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metrics_manager",
        ":utils_impl",
    ],
)

py_library(
    name = "training_loop",
    srcs = ["training_loop.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_manager",
        ":metrics_manager",
        ":utils_impl",
    ],
)

py_test(
    name = "training_loop_test",
    size = "large",
    srcs = ["training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":checkpoint_manager",
        ":metrics_manager",
        ":training_loop",
    ],
)

py_library(
    name = "training_utils",
    srcs = ["training_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "training_utils_test",
    size = "large",
    srcs = ["training_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":training_utils",
    ],
)

py_library(
    name = "threading_utils",
    srcs = ["threading_utils.py"],
    srcs_version = "PY3",
    deps = [":utils_impl"],
)

py_library(
    name = "utils_impl",
    srcs = ["utils_impl.py"],
    srcs_version = "PY3",
)

py_test(
    name = "utils_impl_test",
    srcs = ["utils_impl_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":utils_impl"],
)
