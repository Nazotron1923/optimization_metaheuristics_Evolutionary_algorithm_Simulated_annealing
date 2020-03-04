Metaheuristics (IA-308)
=======================

Compile as PDF: `pandoc -f markdown --toc -o LESSON.pdf LESSON.md`.

Introduction
------------

Metaheuristics are mathematical optimization algorithms solving
$argmin_{x \in X} f(x)$ (or argmax).

Synonyms:

- search heuristics,
- evolutionary algorithms,
- stochastic local search.

The general approach is to only look at the solutions, by trial and error, without further information on its structure.
Hence the problem is often labelled as "black-box".

Link to NP-hardness/curse of dimensionality: easy to evaluate, hard to solve.
Easy to evaluate = fast, but not as fast as the algorithm itself.
Hard to solve, but not impossible.


Algorithmics
------------

Those algorithms are randomized and iteratives (hence stochastics) and manipulates a sample (synonym population)
of solutions (s. individual) to the problem, each one being associated with its quality (s. cost, fitness).

Thus, algorithms have a main loop, and articulate functions that manipulates the sample (called "operators").

Main design problem: exploitation/exploration compromise (s. intensification/diversification).
Main design goal: raise the abstraction level.
Main design tools: learning (s. memory) + heuristics (s. bias).

Forget metaphors and use mathematical descriptions.

Seek a compromise between complexity, performances and explainability.

The is no better "method".
Difference between model and instance, for problem and algorithm.
No Free Lunch Theorem.
But there is a "better algorithm instances on a given problem instances set".

The better you understand it, the better the algorithm will be.


Problem modelization
--------------------

Way to assess the quality: fitness function.
Way to model a solution: encoding.


### Main models

Encoding:

- continuous (s. numeric),
- discrete metric (integers),
- combinatorial (graph, permutation).

Fitness:

- mono-objective,
- multi-modal,
- multi-objectives.


### Constraints management

Main constraints management tools for operators:
- penalization,
- reparation,
- generation.


Performance evaluation
----------------------

### What is performance

Main performances axis:

- time,
- quality,
- probability.

Additional performance axis:

- robustness,
- stability.

Golden rule: the output of a metaheuristic is a distribution, not a solution.


### Empirical evaluation

Proof-reality gap is huge, thus empirical performance evaluation is gold standard.

Empirical evaluation = scientific method.

Basic rules of thumb:

- randomized algorithms => repetition of runs,
- sensitivity to parameters => design of experiments,
- use statistical tools,
- design experiments to answer a single question,
- test one thing at a time.

### Useful statistical tools

Statistical tests:

- classical null hypothesis: test equality of distributions.
- beware of p-value.

How many runs?

- not always "as many as possible",
- maybe "as many as needed",
- generally: 15 (min for non-parametric tests) -- 20 (min for parametric-gaussian tests).

Use robust estimators: median instead of mean, Inter Quartile Range instead of standard deviation.


### Expected Empirical Cumulative Distribution Functions

On Run Time: ERT-ECDF.

$$ERTECDF(\{X_0,\dots,X_i,\dots,X_r\}, \delta, f, t) := \#\{x_t \in X_t | f(x_t^*)>=\delta \}$$

$$\delta \in \left[0, \max_{x \in \mathcal{X}}(f(x))\right]$$

$$X_i := \left\{\left\{ x_0^0, \dots, x_i^j, \dots, x_p^u | p\in[1,\infty[ \right\} | u \in [0,\infty[ \right\} \in \mathcal{X}$$

with $p$ the sample size, $r$ the number of runs, $u$ the number of iterations, $t$ the number of calls to the objective
function.

The number of calls to the objective function is a good estimator of time because it dominates all other times.

The dual of the ERT-ECDF can be easily computed for quality (EQT-ECDF).

3D ERT/EQT-ECDF may be useful for terminal comparison.


### Other tools

Convergence curves: do not forget the golden rule and show distributions:

- quantile boxes,
- violin plots,
- histograms.


Algorithm Design
----------------

### Neighborhood

Convergence definition(s):

- strong,
- weak.

Neighborhood: subset of solutions atteinable after an atomic transformation:

- ergodicity,
- quasi-ergodicity.


### Structure of problem/algorithms

Structure of problems to exploit:

- locality (basin of attraction),
- separability,
- gradient,
- funnels.

Structure with which to capture those structures:

- implicit,
- explicit,
- direct.

Silver rule: choose the algorithmic template that adhere the most to the problem model.

- taking constraints into account,
- iterate between problem/algorithm models.


### Grammar of algorithms

Parameter setting < tuning < control.

Portfolio approaches.
Example: numeric low dimensions => Nelder-Mead Search is sufficient.

Algorithm selection.

Algorithms are templates in which operators are interchangeable.

Most generic way of thinking about algorithms: grammar-based algorithm selection with parameters.
Example: modular CMA-ES.

Parameter setting tools:

- ParamILS,
- SPO,
- i-race.

Design tools:

- ParadisEO,
- jMetal,
- Jenetics,
- ECJ,
- DEAP,
- HeuristicLab.


### Landscape-aware algorithms

Fitness landscapes: structure of problems as seen by an algorithm.
Features: tool that measure one aspect of a fitness landscape.

We can observe landscapes, and learn which algorithm instance solves it better.
Examples: SAT, TSP, BB.

Toward automated solver design.
