# Stanford CS 336: Language Modeling from Scratch

Organic, artisanal, handwritten code for Stanford's CS 336 (Spring 2025) - a course on building language models from the ground up.

**Course Website:** https://stanford-cs336.github.io/spring2025/

## Overview

This repository contains my implementations for three assignments from CS 336, where students build every component of a modern language model with minimal scaffolding. Each assignment took maybe 40 hours and cost under $100 to complete (mostly for rented gpu compute).

## Assignments

### Assignment 1: Basics
**Implementation:** [`assignment1-basics/`](./assignment1-basics/) | **Writeup:** [`writeup1.pdf`](./writeup1.pdf)

Build a Transformer language model from scratch using PyTorch primitives - BPE tokenizer, multi-head attention with RoPE, SwiGLU, RMSNorm, and AdamW.

### Assignment 2: Systems
**Implementation:** [`assignment2-systems/`](./assignment2-systems/) | **Writeup:** [`writeup2.pdf`](./writeup2.pdf)

Optimize Transformer performance through Flash Attention 2 in Triton, DDP training, optimizer state sharding, and custom fused kernels.

### Assignment 5: Alignment
**Implementation:** [`assignment5-alignment/`](./assignment5-alignment/)

Align language models on math dataset using supervised fine-tuning, expert iteration, and GRPO on the Qwen 2.5 Math 1.5B model.
Note: the math dataset that they use is copyrighted but can be found on huggingface.

## Acknowledgments

Thank you very much to the Stanford CS336 teaching team. Course materials and starter code provided by them on their github. All code are implementations following the assignment specs.