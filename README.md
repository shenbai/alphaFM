# alphaFM
## forked from https://github.com/CastellanZhang/alphaFM

与原版不同之处在于，模型限定了特征空间的大小。默认为2^28，特征需要提前hash化，将hash code与特征空间取余。
同时不再采用map结构，改用vector，通过下标直接定位特征，提高了速度同时降低了内存消耗。
