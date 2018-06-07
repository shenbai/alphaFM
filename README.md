# alphaFM
## forked from https://github.com/CastellanZhang/alphaFM

```
与原版不同之处在于：
1模型限定了特征空间的大小。默认为2^28，特征需要提前hash化，将hash code与特征空间取余。
2不再采用map结构，改用vector，通过下标直接定位特征，提高了速度同时降低了内存消耗。
3模型输出时忽略w和v均为0的参数，减少输出模型的大小。
```

```
usage: cat sample | ./fm_train [<options>]

options:
-m <model_path>: set the output model path
-s <space_size>: set space size, default 2^28
-compress <compress>: compress when output model, default 0
-dim <k0,k1,k2>: k0=use bias, k1=use 1-way interactions, k2=dim of 2-way interactions	default:1,1,8
-init_stdev <stdev>: stdev for initialization of 2-way factors	default:0.1
-w_alpha <w_alpha>: w is updated via FTRL, alpha is one of the learning rate parameters	default:0.05
-w_beta <w_beta>: w is updated via FTRL, beta is one of the learning rate parameters	default:1.0
-w_l1 <w_L1_reg>: L1 regularization parameter of w	default:0.1
-w_l2 <w_L2_reg>: L2 regularization parameter of w	default:5.0
-v_alpha <v_alpha>: v is updated via FTRL, alpha is one of the learning rate parameters	default:0.05
-v_beta <v_beta>: v is updated via FTRL, beta is one of the learning rate parameters	default:1.0
-v_l1 <v_L1_reg>: L1 regularization parameter of v	default:0.1
-v_l2 <v_L2_reg>: L2 regularization parameter of v	default:5.0
-core <threads_num>: set the number of threads	default:1
-im <initial_model_path>: set the initial value of model
-fvs <force_v_sparse>: if fvs is 1, set vi = 0 whenever wi = 0	default:0
```
