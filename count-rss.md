# Quantifying reasoning shortcuts: `count-rss`

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
