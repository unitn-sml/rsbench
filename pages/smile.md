---
layout: default
title: SMiLe Poster Presentation
permalink: /smile.html
---

# SMiLe Poster Presentation

A shorthand for the links to the papers presented at the SMiLe workshop

---

<h2><a name="papers">Papers</a></h2>


- [`bears` Make Neuro-Symbolic Models Aware of their Reasoning Shortcuts](https://proceedings.mlr.press/v244/marconato24a.html)  
  _Neuro-Symbolic (NeSy) predictors that conform to symbolic knowledge - encoding, e.g., safety constraints - can be affected by Reasoning Shortcuts (RSs): They learn concepts consistent with the symbolic knowledge by exploiting unintended semantics. RSs compromise reliability and generalization and, as we show in this paper, they are linked to NeSy models being overconfident about the predicted concepts. Unfortunately, the only trustworthy mitigation strategy requires collecting costly dense supervision over the concepts. Rather than attempting to avoid RSs altogether, we propose to ensure NeSy models are aware of the semantic ambiguity of the concepts they learn, thus enabling their users to identify and distrust low-quality concepts. Starting from three simple desiderata, we derive bears (BE Aware of Reasoning Shortcuts), an ensembling technique that calibrates the model's concept-level confidence without compromising prediction accuracy, thus encouraging NeSy architectures to be uncertain about concepts affected by RSs. We show empirically that bears improves RS-awareness of several state-of-the-art NeSy models, and also facilitates acquiring informative dense annotations for mitigation purposes._

- [A Neuro-Symbolic Benchmark Suite for Concept Quality and Reasoning Shortcuts](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d1d11bf8299334d354949ba8738e8301-Abstract-Datasets_and_Benchmarks_Track.html)  
  _The advent of powerful neural classifiers has increased interest in problems that require both learning and reasoning. These problems are critical for understanding important properties of models, such as trustworthiness, generalization, interpretability, and compliance to safety and structural constraints. However, recent research observed that tasks requiring both learning and reasoning on background knowledge often suffer from reasoning shortcuts (RSs): predictors can solve the downstream reasoning task without associating the correct concepts to the high-dimensional data. To address this issue, we introduce rsbench, a comprehensive benchmark suite designed to systematically evaluate the impact of RSs on models by providing easy access to highly customizable tasks affected by RSs. Furthermore, rsbench implements common metrics for evaluating concept quality and introduces novel formal verification procedures for assessing the presence of RSs in learning tasks. Using rsbench, we highlight that obtaining high quality concepts in both purely neural and neuro-symbolic models is a far-from-solved problem. rsbench is available at: [this https URL](https://unitn-sml.github.io/rsbench/)._

- [Shortcuts and Identifiability in Concept-based Models from a Neuro-Symbolic Lens](https://arxiv.org/abs/2502.11245)  
  _Concept-based Models are neural networks that learn a concept extractor to map inputs to high-level concepts and an inference layer to translate these into predictions. Ensuring these modules produce interpretable concepts and behave reliably in out-of-distribution is crucial, yet the conditions for achieving this remain unclear. We study this problem by establishing a novel connection between Concept-based Models and reasoning shortcuts (RSs), a common issue where models achieve high accuracy by learning low-quality concepts, even when the inference layer is fixed and provided upfront. Specifically, we first extend RSs to the more complex setting of Concept-based Models and then derive theoretical conditions for identifying both the concepts and the inference layer. Our empirical results highlight the impact of reasoning shortcuts and show that existing methods, even when combined with multiple natural mitigation strategies, often fail to meet these conditions in practice._

---

<h2><a name="codebase">Codebase</a></h2>

- [`bears`](https://github.com/samuelebortolotti/bears)  
  _Implements methods and experiments described in [`bears` Make Neuro-Symbolic Models Aware of their Reasoning Shortcuts](https://proceedings.mlr.press/v244/marconato24a.html)._

- [`rsbench`](https://github.com/unitn-sml/rsbench-code)  
  _Implements methods and experiments described in [A Neuro-Symbolic Benchmark Suite for Concept Quality and Reasoning Shortcuts](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d1d11bf8299334d354949ba8738e8301-Abstract-Datasets_and_Benchmarks_Track.html) and [Shortcuts and Identifiability in Concept-based Models from a Neuro-Symbolic Lens](https://arxiv.org/abs/2502.11245)._

---