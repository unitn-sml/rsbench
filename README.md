# A Neuro-Symbolic Benchmark Suite for Concept Quality and Reasoning Shortcuts

Web site for the ["A Neuro-Symbolic Benchmark Suite for Concept Quality and Reasoning Shortcuts"](https://arxiv.org/abs/2406.10368) paper made with [Jekyll](https://jekyllrb.com/) and hosted with [GitHub Pages](https://pages.github.com/), [NeurIPS 2024](https://neurips.cc/Conferences/2024).

Apart from MathJax for LaTeX formula rendering and the dark-light mode toggle, the website is completely static and does not include JavaScript.

## Github Pages deployment

If you do not have it yet, install [Ruby](https://www.ruby-lang.org/en/). To install the latest version you can call:

```bash
make install-ruby
```

Install the Ruby gems with [Bundler](https://bundler.io/):

```bash
bundle install
```

or alternatively:

```bash
make install
```

> For other functionalities you can check out the [Makefile](Makefile) rules by typing:
> ```bash
> make help
> ```

# Codebase

The codebase for generating the datasets, evaluating them, and counting the Reasoning Shortcuts is available at the following link: [GitHub repository](https://github.com/unitn-sml/rsbench-code).