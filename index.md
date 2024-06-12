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


<h1 class="heading blue"><a name="downloads">Downloads</a></h1>

### **Data**: [GDrive](https://todo), [Zenodo](https://todo)

### **Preprint**: [ArXiV](https://arxiv.org)


<h1 class="heading blue"><a name="background">What is a Reasoning Shortcut?</a></h1>

<img src="assets/images/reasoning-shortcut.png" alt="a reasoning shortcut" width="100%" height="auto">

**What are L&R tasks?**  In learning and reasoning tasks, machine learning
models should predict labels that comply with prior knowledge.  For instance,
in autonomous vehicle scenario, the model should predict `stop` or `go` based
on what obstacles are visible in front of the vehicle, and the prior knowledge
encodes the rule that if a `pedestrian` or a `red_light` is visible then it
should definitely predict `stop`.

**What is a reasoning shortcut?**  A RS occurs when the model predicts the
right label by inferring the wrong concepts.  For instance, it might confuse
`pedestrian`s for `red_light`s as both entail the same (correct) `stop` action.

**What are the consequences?** RSs can compromise the *interpretability* of
model explanations (e.g., these might show that a prediction depends on the
`red_light`s present in the image, while in reality it depends on
`pedestrian`s!) and *generalization* to out-of-distribution tasks (e.g., if a
vehicle is authorized to cross over `red_light`s in the case of an emergency,
and it confuses these with `pedestrian`s, this might lead to harmful
decisions).

<span style="font-size:0.5em;">Image taken with permission from: Marconato *et
al.* "Not all neuro-symbolic concepts are created equal: Analysis and
mitigation of reasoning shortcuts." NeurIPS 2023.</span>


<h1 class="heading blue"><a name="overview">Overview</a></h1>

- *A Variety of L&R Tasks*: rsbench offers five L&R tasks and at least one data
  set each.  The tasks come in differet flavous: arithmetic, logic, and
  high-stakes.  It also provides data generators for creating new OOD splits
  useful for testing the down-stream consequences of RSs.

- *Evaluation*: rsbench comes with implementations for several metrics for
  evaluating the quality of *label* and *concept* predictions, as well as
  visualization code for them.

- *Verification*: rsbench implements a new algorithm, `countrss`, that makes
  use of automated reasoning packages for formally veryfing whether a L&R task
  allows for RSs without training any model!  This tool works with any prior
  knowledge encoded in CNF format, the de-facto standard in automated
  reasoning.

- *Example code*: our repository comes with example code for training and
  evaluating a selection of state-of-the-art machine learning architectures,
  including Neuro-Symbolic models, Concept-bottleneck models, and regular
  neural networks.


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


<h1 class="heading blue"><a name="usage">Usage</a></h1>

TODO: how to train a NeSy model and how to evaluate performance.  link to jupyter notebook?


# MNISTMath

<img src="assets/images/rsbench-mnmath.png" alt="mnmath" width="80%" height="auto">

WRITEME

**Ready-made**: `MNAdd-Half` WRITEME

**Ready-made**: `MNAdd-EvenOdd` WRITEME


# MNISTLogic

<img src="assets/images/rsbench-mnmath.png" alt="mnmath" width="80%" height="auto">

WRITEME


# Kand-Logic

<img src="assets/images/rsbench-kandlogic.png" alt="mnmath" width="80%" height="auto">

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


<h1 class="heading blue"><a name="evaluation">Evaluation</a></h1>

WRITEME


<h1 class="heading blue"><a name="verification">Verification</a></h1>

`count-rss` is a small tool that is able to enumerate the RSs in a task by reducing the task to model counting (\#SAT).
Due to their large number even on seemingly simple tasks, `count-rss` includes the state-of-the-art approximate \#SAT solver `ApproxMC` (https://github.com/meelgroup/approxmc).
Custom tasks can be defined as logical formulas in conjunctive normal form (CNF) in DIMACS format. 
The output of the encoding procedure is itself a DIMACS file.

# Usage

## Generating the RSs counting encoding

Use `python gen-rss-count.py` for generating a DIMACS encoding of the counting task.

On small datasets/tasks, the count of RSs can be computed directly (and exactly) with the `-E` flag.
For instance:
```
$ python gen-rss-count.py xor -n 3 -E
```
computes all the RSs resulting from the XOR task on 3 variables with exhaustive supervision.

Partial/incomplete supervision can be controlled with `-d P` with P in [0,1].
For instance:
```
$ python gen-rss-count.py xor -n 3 -E -d 0.25
```
computes all the RSs when only 1/4 (i.e. 2 examples) are provided.
The optional `--seed`  argument sets the seed number.

Beyond illustrative the XOR case, random CNFs with N variables, M clauses of length K can be evaluated:
```
$ python gen-rss-count.py random -n N -m M -k K
```

Custom task expressed in DIMACS format are supported, for instance:
```
$ python gen-rss-count.py cnf and.cnf
```

Use the flag `-h` for help on additional arguments.

## Approximately count RSs via approximate model counting

Once the encoding of the problem is generated with `gen-rss-count.py`, use:
```
$ python count-amc.py PATH --epsilon E --delta D
```
for obtaining an (epsilon,delta)-approximation of the exact RS count.

<h1 class="heading blue"><a name="license">License</a></h1>

TODO: @Samuele, copy from the paper
