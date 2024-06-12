---
layout: default
title: rsbench: A Benchmark Suite for Systematically Evaluating Reasoning Shortcuts
---

{% include header.html %}

# Abstract

The advent of powerful neural classifiers has increased interest in problems
that require both learning and reasoning. These problems are critical for
understanding important properties of models, such as trustworthiness,
generalization, interpretability, and compliance to safety and structural
constraints. However, recent research observed that tasks requiring both
learning and reasoning on background knowledge often suffer from reasoning
shortcuts (RSs): predictors can solve the downstream reasoning task without
associating the correct concepts to the high-dimensional data. To address this
issue, we introduce rsbench, a comprehensive benchmark suite designed to
systematically evaluate the impact of RSs on models by providing easy access to
highly customizable tasks affected by RSs. Furthermore, rsbench implements
common metrics for evaluating concept quality and introduces novel formal
verification procedures for assessing the presence of RSs in learning tasks.
Using rsbench, we highlight that obtaining high quality concepts in both purely
neural and neuro-symbolic models is a far-from-solved problem.


# Links

### **Data**: [GDrive](https://todo), [Zenodo](https://todo)

### **Preprint**: [ArXiV](https://arxiv.org)


# What is a Reasoning Shortcut?

TODO: add BEARS or NeurIPS figure 2.

![test](assets/images/logo-sml.png)

**What are L&R tasks?**  In learning and reasoning tasks, machine learning
models should predict labels that comply with prior knowledge.  For instance,
in autonomous vehicle scenario, the model should predict `stop` or `go` based
on what obstacles are visible in front of the vehicle, and the prior knowledge
encodes the rule that if a `pedestrian` or a `red_light` is visible then it
should definitely predict `stop`.

**What is a reasoning shortcut?**  A RS occurs when the model predicts the
right label by inferring the wrong concepts.  For instance, it might confuse
`pedestrian`s for `red_light`s as both entail the same (correct) `stop` action.

**What consequences to RSs have?** RSs can compromise model explanations (e.g.,
because these show that `red_light`s are responsible for the predictions, while
in fact this depends on the presence of red lights


# Key Features

- *A Variety of L&R Tasks*: WRITEME different types of input and flavours of
  knowledge.  Support for OOD splits.

- *Evaluation*:

- *Verification*:

- *Example code*:


# Overview

rsbench supplies several *data sets* for 5 learning and reasoning (L&R) tasks.
It also provides *data generators* for creating additional data splits.  


| L&R Task        | Images       | Concepts                                                             | Labels                      | #Train | #Valid | #Test  | #OOD   |
| :--             | :--:         | :--:                                                                 | :--:                        | :--:   | :--:   | :--:   | :--:   |
| `MNMath`        | 28k x 28     | k digits, 10 values each                                             | categorical multilabel      | custom | custom | custom | custom |
| `MNAdd-Half`    | 56 x 28      | 2 digits, 10 values each                                             | categorical 0 ... 18        | 2,940  | 840    | 420    | 1,080  |
| `MNAdd-EvenOdd` | 56 x 28      | 2 digits, 10 values each                                             | categorical 0 ... 18        | 6,720  | 1,920  | 960    | 5,040  |
| `MNLogic`       | 28k x 28     | k digits, 10 values each                                             | binary                      | custom | custom | custom | custom |
| `Kand-Logic`    | 3 x 192 x 64 | 3 objects per image, 3 shapes, 3 colors                              | binary                      | 4,000  | 1,000  | 1,000  | -      |
| `CLE4EVR`       | 320 x 240    | n to m objects per image, 10 shapes, 10 colors, 2 materials, 3 sizes | binary                      | custom | custom | custom | custom |
| `BDD-OIA`       | 1280 x 720   | 21 binary concepts                                                   | binary multilabel, 4 labels | 16,082 | 2,270  | 4,572  | --     |
| `SDD-OIA`       | 469 x 387    | 21 binary concepts                                                   | binary multilabel, 4 labels | 6,820  | 1,464  | 1,464  | 1,000  |


# How To Use rsbench

TODO: how to train a NeSy model and how to evaluate performance.  link to jupyter notebook?


# MNISTMath

TODO: add figure

WRITEME

**Ready-made**: `MNAdd-Half` WRITEME

**Ready-made**: `MNAdd-EvenOdd` WRITEME


# MNISTLogic

TODO: add figure

WRITEME


# Kand-Logic

TODO: add figure

WRITEME


# CLE4EVR

TODO: add figure

WRITEME


# BDD-OIA

TODO: add figure

WRITEME


# SDD-OIA

TODO: add figure

WRITEME


# Evaluation

WRITEME


# Verification

TODO: @Paolo
