# Federated Composite Optimization
This code repository is for "[Federated Composite Optimization](https://arxiv.org/abs/2011.08474)" authored by [Honglin Yuan](https://hongliny.github.io) (Stanford), Manzil Zaheer (Google) and Sashank Reddi (Google), to appear in ICML 2021. 

[talk video at FLOW seminar](https://www.youtube.com/watch?v=tKDbc60XJks) 
| [slides (pdf)](https://hongliny.github.io/files/FCO_ICML21/FCO_ICML21_slides.pdf) 

bibtex
```
@inproceedings{DBLP:conf/icml/YuanZR21,
  author    = {Honglin Yuan and
               Manzil Zaheer and
               Sashank J. Reddi},
  editor    = {Marina Meila and
               Tong Zhang},
  title     = {Federated Composite Optimization},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning,
               {ICML} 2021, 18-24 July 2021, Virtual Event},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {12253--12266},
  publisher = {{PMLR}},
  year      = {2021},
  url       = {http://proceedings.mlr.press/v139/yuan21d.html},
  timestamp = {Wed, 14 Jul 2021 15:41:58 +0200},
  biburl    = {https://dblp.org/rec/conf/icml/YuanZR21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

---

This repo is forked from [Federated Research](https://github.com/google-research/federated) repository developed with [TensorFlow Federated (TFF)](https://www.tensorflow.org/federated), an open-source framework for machine learning and other computations on decentralized data.

Some pip packages are required by this library, and may need to be installed:

```
pip install absl-py
pip install attr
pip install dm-tree
pip install numpy
pip install pandas
pip install tensorflow
pip install tensorflow-federated
```

We also require [Bazel](https://www.bazel.build/) in order to run the code.
Please see the guide
[here](https://docs.bazel.build/versions/master/install.html) for installation
instructions.

## Directory structure

The directory `utils` is forked from [Federated Research](https://github.com/google-research/federated) repository, which contains general utilities used by the other directories under research/. Examples include utilities for saving checkpoints, and configuring experiments via command-line flags. 

The directory `optimization` is broken up into five task directories. Each task directory contains task-specific libraries (such as libraries for loading the correct dataset), as well as libraries for performing federated and non-federated (centralized) training. These are in the `optimization/{task}` folders.

A single binary for running these tasks can be found at `main/federated_trainer.py`. This binary will, according to `absl` flags, run any of the six task-specific federated training libraries.

The `optimization/shared` directory with utilities specific to these experiments, such as implementations of metrics used for evaluation.
