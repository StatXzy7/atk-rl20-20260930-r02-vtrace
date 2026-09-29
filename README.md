# V-trace actor–learner 策略滞后实验

这是一个可精确核验的有限 MDP V-trace 实验。全部核心 off-policy correction、价值目标和策略梯度均为手写，不依赖任何现成 V-trace/IMPALA 实现。

## 快速开始

```bash
pytest -q
python scripts/run_experiment.py --seeds 12 --episodes-per-seed 200
```

结果写入 `results/vtrace_experiment.json`。脚本不下载数据，也不使用预训练模型。

## 文件结构

- `src/vtrace_lab/mdp.py`：有限状态、有限动作、已知转移概率和有界奖励 MDP。
- `src/vtrace_lab/sampling.py`：显式版本号的滞后 actor；每条 transition 保存实际采样动作概率、完整行为分布、actor 版本和采样时 learner 版本。
- `src/vtrace_lab/vtrace.py`：逐轨迹参考版和 padding 批量版 V-trace；`rho_bar`、`c_bar` 独立配置。
- `src/vtrace_lab/exact.py`：用线性方程精确求 `V^mu`、`V^pi`、截断修正策略价值和期望 V-trace 算子。
- `src/vtrace_lab/learning.py`：手写表格 softmax 策略梯度和价值梯度，优势和重要性比率作为无梯度标量数据使用。
- `src/vtrace_lab/experiments.py`：π=μ、无截断、极端滞后、支持集缺失、c 权衡和版本化 actor–learner 更新实验。
- `tests/`：精确 Bellman 残差、padding 等价、零行为概率、终止/截断、固定点和梯度有限差分测试。

## V-trace 递推

对轨迹时间步 `t`，记：

- `r_t`：有界奖励；
- `a_t`：实际执行动作；
- `mu(a_t|s_t)`：采样时真实行为概率；
- `pi(a_t|s_t)`：采样时 learner 目标策略对同一动作给出的概率；
- `rho_t = min(rho_bar, pi(a_t|s_t)/mu(a_t|s_t))`；
- `c_t = min(c_bar, pi(a_t|s_t)/mu(a_t|s_t))`；
- `V_t`：当前 learner 对 `s_t` 的价值估计；
- `b_T`：截断轨迹末端的 bootstrap 价值；真实终止时为 0。

反向 trace 为：

```text
delta_t = rho_t (r_t + gamma V_{t+1} - V_t)
trace_t = delta_t + gamma c_t trace_{t+1}
v_t^Vtrace = V_t + trace_t
```

在轨迹边界：

- 真实终止：下一价值为 `0`，因为终止态奖励和价值均为零；
- 仅采样截断：下一价值为记录的 `V(s_T)` bootstrap；
- padding：不参与递推，输出保持 `NaN`。

因此，`rho_t` 截断当前状态的局部 TD 修正；`c_t` 只截断未来修正向当前状态传播的强度。二者必须分开。代码强制 `0 < c_bar <= rho_bar`，与 V-trace 常用约定一致。

## 固定点对应哪个策略

设状态内定义：

```text
z(s)   = E_mu[min(rho_bar, pi(A|s)/mu(A|s))]
       = sum_a min(pi(a|s), rho_bar mu(a|s))
pi_rho(a|s) = min(pi(a|s), rho_bar mu(a|s)) / z(s)
```

若 `z(s)>0`，则：

```text
E_mu[rho_t (r + gamma V(S') - V(s))]
  = z(s) (r^{pi_rho}(s) + gamma P^{pi_rho} V(s) - V(s))
```

`c` 只出现在路径传播项中。当状态在行为策略下可持续访问时，无限视野期望 V-trace 算子的固定点满足截断策略 `pi_rho` 的 Bellman 方程：

```text
V(s) = r^{pi_rho}(s) + gamma sum_{s'} P^{pi_rho}(s,s') V(s')
```

所以代码用线性方程 `(I - gamma P^{pi_rho}) V = r^{pi_rho}` 精确求固定点。

当 `rho_bar -> infinity` 且 `pi` 对所有 `mu` 可采样动作都有正概率时，`min(pi, rho_bar mu)=pi`，于是 `pi_rho=pi`，固定点才是原始目标策略价值 `V^pi`。

有限 `rho_bar` 通常改变状态动作权重：高 `pi/mu` 动作的贡献被压到 `rho_bar mu`，然后重新归一化。因此即使采样完全无噪声，固定点也通常不是 `V^pi`，这是算法目标偏差，不是有限样本误差。

## c_bar 的作用

`c_bar` 不改变每个状态内由 `rho_bar` 决定的固定点策略。它控制未来 TD 修正沿行为策略路径回传多少：

- `c_bar` 越小：远处修正被更快衰减，通常降低轨迹回报方差，但算子收缩更强、到达同一固定点所需迭代更多；
- `c_bar` 越大：保留更多多步信息，固定点迭代收敛更快，但重要性权重乘积会带来更大的方差；
- 在有限样本下，bootstrap、轨迹长度和随机访问分布仍会让不同 `c_bar` 的估计不同；固定点相同不等于有限样本估计相同。

实验 JSON 同时报告 `c_fixed_point_and_convergence` 和 `initial_estimate_variance_within_seeds`。

## 支持集缺失

若 `pi(a|s)>0` 但 `mu(a|s)=0`，则 `pi/mu` 是正无穷而非“很大但有限”。给 `mu` 加 epsilon 会改变真实行为策略，不能据此宣称已经纠偏。

本实现：

- 保存的是未加 epsilon 的真实行为概率；
- 对正 `pi`、零 `mu` 的比率直接抛出 `ZeroDivisionError`，除非显式请求 `NaN`；
- `clipped_policy` 会丢弃这些不可采样动作的质量并重新归一化；
- `missing_support_one_shot` 场景报告缺失质量；未观测动作的 `V^pi` 部分不能由该行为策略的数据恢复。

## 策略梯度

与 V-trace 目标一致的价值优势是：

```text
A_t = v_t^Vtrace - V(s_t)
```

策略梯度使用当前动作的额外 `rho_t`：

```text
g = sum_t rho_t A_t grad_theta log pi_theta(a_t|s_t)
```

代码中分别保存为 `advantages` 和 `policy_gradient_advantages`。策略更新只通过 `log pi_theta` 接收梯度；目标概率、行为概率、比率、target 和 advantage 都是外部固定标量。Numpy 实现没有 autograd 图，因此不会意外把梯度传入行为概率。

## 明确反例：降低方差但引入目标偏差

`make_one_shot_mdp` 只有一个非终止状态：

- 奖励：动作 0 为 `+1`，动作 1 为 `-1`，一步终止；
- `pi = [0.8, 0.2]`，故 `V^pi = 0.6`；
- `mu = [0.2, 0.8]`；
- `rho_bar=c_bar=1` 时，`pi_rho=[0.5,0.5]`，故 `V^{pi_rho}=0.0`。

无截断 IS 估计以 `0.2/0.8` 的概率得到重要性比率 4，估计固定点仍为 `0.6`，但单样本方差大。截断后比率分别变成 1 和 0.25，方差下降，但估计收敛到 `0.0`；相对原始 `pi` 的固定点偏差为 `-0.6`。

## 误差口径

每个场景都报告：

- 多种子误差均值；
- 每种子实际 episode 数和 transition 数；
- 真实终止数与仅采样截断数；
- 状态访问计数；
- 相对 `V^pi` 的 RMSE；
- 相对精确 `V^{pi_rho}` 的 RMSE。

若估计接近 `V^{pi_rho}` 而不接近 `V^pi`，这是固定点偏差；若接近理论固定点但仍有种子间波动，这是有限样本估计误差。
