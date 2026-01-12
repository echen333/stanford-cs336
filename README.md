# CS336 Spring 2025 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment2_systems.pdf](./cs336_2025spring_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
❯ cd ..
❯ cat README.md
# Stanford CS 336: Language Modeling from Scratch

Organic, artisanal, handwritten code for Stanford's CS 336 (Spring 2025) - a course on building language models from the ground up.

**Course Website:** https://stanford-cs336.github.io/spring2025/

## Overview

This repository contains my implementations for three assignments from CS 336, where students build every component of a modern language model with minimal scaffolding. Each assignment took maybe 40-80 hours and cost under $100 to complete (mostly for rented gpu compute).

Most of my thoughts align with [andytimm](https://andytimm.github.io/posts/cs336/cs336_review.html). You should be very proficient in python, and assignments 1,2, and 5 definitely take a lot of time and . Personally, I chose to not use any LLM's to write code but did use it a lot to brainstorm.

## Assignments

### Assignment 1: Basics
**Implementation:** [`assignment1-basics/`](./assignment1-basics/) | **Writeup:** [`writeup1.pdf`](./writeup1.pdf)

Build a Transformer language model from scratch using PyTorch primitives - BPE tokenizer, multi-head attention with RoPE, SwiGLU, RMSNorm, and AdamW.

### Assignment 2: Systems
**Implementation:** [`assignment2-systems/`](./assignment2-systems/) | **Writeup:** [`writeup2.pdf`](./writeup2.pdf)

Optimize Transformer performance through Flash Attention 2 in Triton, DDP training, optimizer state sharding, and custom fused kernels.

### Assignment 3: Scaling Laws (Incomplete)

Unfortunately, the bulk of assignment 3 relies on an API only available to Stanford CS336 students.

### Assignmnet 4: Data (Incomplete)

Filtering web crawls to create data for language modeling.

### Assignment 5: Alignment
**Implementation:** [`assignment5-alignment/`](./assignment5-alignment/)

Align language models on math dataset using supervised fine-tuning, expert iteration, and GRPO on the Qwen 2.5 Math 1.5B model.

## Acknowledgments

Thank you very much to the Stanford CS336 teaching team. Course materials and starter code provided by them on their github.
