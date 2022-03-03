# Learning Dynamics Models with Stable Invariant Sets

Implementation of the method presented in the following paper:

Naoya Takeishi and Yoshinobu Kawahara.
Learning dynamics models with stable invariant sets.
In *Proceedings of the 35th AAAI Conference on Artificial Intelligence* (AAAI), pages 9782–9790, 2021.

[https://ojs.aaai.org/index.php/AAAI/article/view/17176](https://ojs.aaai.org/index.php/AAAI/article/view/17176)
[https://arxiv.org/abs/2006.08935](https://arxiv.org/abs/2006.08935)

## Prerequisite

- scipy
- numpy
- matplotlib
- pytorch 1.4.0 or later
- cvxpylayers
- torchdiffeq

## Usage

### General

Use `train.py` to train a dynamics model with a stable invariant set. See the following examples for detailed configurations.

### Experiment of the simple example of limit cycle

For training, execute `do_train.sh` in `toy_limcyc` directory. Then, execute `do_test.sh traj`, `do_test.sh vf`, or `do_test.sh test` to examine the test results. The figures in the paper were created using pgfplots.

### Experiment of the simple example of equilibria set

For training, execute `do_train.sh` in `toy_staeq` directory. Then, execute `do_test.sh` to examine the learned V function.

### Experiment of nonlinear oscillator

For training, execute `do_train.sh` in `nlinosc` directory. Then, execute `do_test.sh lyap` to examine the learned V function.

### Experiment of fluid flow

For training, execute `do_train.sh` in `flow` directory. Then, execute `do_test.sh` to perform long-term prediction. Finally, execute `do_reduct.m` (with MATLAB) to examine the prediction results. The final prediction plots are saved in `reduct` directory. We did not attach the original flow data as it was too large.

## Author

*  **Naoya Takeishi** - [https://ntake.jp/](https://ntake.jp/)
