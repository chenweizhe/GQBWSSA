# GQBWSSA IDS Feature Selection
Paper: Large-scale IoT attack detection scheme based on LightGBM and feature selection using an improved salp swarm algorithm

原项目是在matlab上做的，为了方便复现 目前移植到python中，适应度函数中的随机森林用 CuPy/CuML 在显卡上训练和预测，比matlab训练要快点，对比算法为 GWO、WOA、DBO、HHO。

---




## 1. 快速开始

在 `GQBWSSA_python` 目录下执行：

```powershell
# 只跑 GQBWSSA
python main.py

# 指定算法、重复次数和随机种子
python main.py --algos GWO,WOA,GQBWSSA --runs 5 --seed 46

```



---

## 2. 适应度函数

与 `Fitness_RF.m` 一致：

```
fit = 0.99 × (1 − ACC) + 0.01 × 所选特征数 / (dim + 1)
```


---


## 3. 与 MATLAB 版本的其他差异

- **随机数**：部分逐元素的随机数循环改成了整行一次生成，分布完全相同，但随机序列不同，所以结果不会与 MATLAB 逐位一致（Matlab版本从数据划分开始都为完全随机）；
- **HHO**：原代码在同一分支中对同一个体 `X(i,:)` 计算了两次适应度，Python 版计算一次后复用，少训练一次随机森林；
- **四舍五入**：GQBWSSA 中的 `round` 使用 MATLAB 的“0.5 远离 0”规则，而不是 numpy 默认的银行家舍入；
- Python 和 matlab的数值精度可能会有所不同

除上述几点外，各算法的更新公式、参数（如 GWO 中 Beta 狼的系数 5、GQBWSSA 中的惯性权重和步长、DBO 的四类蜣螂比例）均按原代码保留。

---

