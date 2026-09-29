# Actor–Learner 策略滞后下的 V-trace 精确核验

这个工程在有限状态、有限动作、已知转移概率、有界奖励的折扣 MDP 中手工实现 V-trace，不使用任何现成 off-policy correction 库。采样使用确定性的、可配置版本滞后的 actor；learner 只读取保存下来的行为概率和策略版本。

## 快速运行

```bash
python -m pytest -q
python -m vtrace_lab.experiment --seeds 20 --output results/experiment.json
```

不下载数据，也不加载预训练模型。实验结果会同时打印并写入 JSON。

## MDP 与采样

- `vtrace_lab/mdp.py`：显式吸收终止状态的有限 MDP，奖励必须有限，转移概率逐行归一化。
- `vtrace_lab/scenarios.py`：策略滞后链、支持集缺口、偏差/方差反例和 `c` 传播链。
- `vtrace_lab/sampler.py`：冻结一个旧版本策略作为 actor，逐 transition 保存实际 `μ(a|s)`、采样时 `π_version(a|s)`、版本号、真实终止、仅截断和真实样本长度。

真实终止 `done=True` 时下一价值是 `0`；仅采样截断 `truncated=True` 时使用实际下一状态的 bootstrap value。终止后的位置是 padding，`valid=False`，不进入目标、损失或实际样本数。

## 手工 V-trace 递推

定义行为策略与目标策略的真实概率：

\[
r_t(a_t)=\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)},\qquad
\rho_t=\min(\bar\rho,r_t),\qquad
c_t=\min(\bar c,r_t).
\]

约束为：

\[
0<\bar c\le \bar\rho.
\]

代码从不给分母加 epsilon。若某采样动作满足 `pi_prob > 0` 且 `behavior_prob == 0`，NumPy 和 PyTorch 实现都直接报错。`0/0` 只表示目标和行为都不选择该动作，定义为零权重。

反向递推为：

\[
\delta_t=\rho_t(r_t^{\mathrm{env}}+\gamma V(s_{t+1})-V(s_t)),
\]

\[
A_t=\delta_t+\gamma c_t A_{t+1},
\qquad
V_s^{target}(s_t)=V(s_t)+A_t.
\]

边界为：

- `done_t=True`：`V(s_{t+1})=0`，`A_{t+1}=0`；
- segment 在非终止状态结束：`V(s_{t+1})=bootstrap`，`A_{t+1}=0`；
- padding：完全 mask，不产生递推项。

`ρ` 出现在一步 Bellman 误差内部，决定哪些动作的 TD 误差进入目标；`c` 只出现在后续状态 advantage 的时间乘子上，决定未来校正沿一条样本轨迹传播多远。

## 固定点对应哪个策略

在行为策略数据下取期望，一步校正残差为：

\[
g_s(V)=\sum_a \mu(a\mid s)\rho(a\mid s)
\left[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')-V(s)\right].
\]

无限 horizon 固定点满足 `g_s(V)=0`。定义未归一化的截断动作权重：

\[
w^{\bar\rho}(a\mid s)=\min\bigl(\pi(a\mid s),\bar\rho\mu(a\mid s)\bigr).
\]

当 `w` 在每个非终止状态都有正总质量时，固定点策略是：

\[
\pi_{\bar\rho}(a\mid s)=
\frac{w^{\bar\rho}(a\mid s)}
{\sum_b w^{\bar\rho}(b\mid s)}.
\]

`vtrace_lab/exact.py:solve_vtrace_fixed` 直接构造并求解这个线性方程；`corrected_policy` 构造上述归一化策略，再用普通 Bellman 线性方程求其价值。二者在测试中数值一致。

有限 `ρ_bar` 通常不等于原策略 `π`：

- 若 `π/μ < ρ_bar`，动作保持目标权重；
- 若 `π/μ > ρ_bar`，权重被压到 `ρ_bar μ`；
- 归一化后，高重要性比率动作的质量被重新分配给其他被 μ 支持的动作。

只有在完全支持集且取消 `ρ` 截断（`ρ_bar=∞`）时，期望固定点才回到原始 `π`。`π=μ` 时所有比率都是 1，任意允许的截断都不改变固定点。

## 支持集缺口

若对某动作有 `π(a|s)>0` 且 `μ(a|s)=0`：

\[
w^{\bar\rho}(a\mid s)=0,
\]

该动作永远不会从 μ 数据中获得非零的有限样本权重。若某个状态的所有权重都为零，固定点线性系统奇异；这不是数值稳定问题，给 μ 分母加 epsilon 只是在虚构行为概率，不能声称完成纠偏。

支持集缺口场景在 `results/experiment.json` 的 `support_gap` 中报告缺失状态-动作、零截断权重和奇异线性系统。

## c 截断：固定点、方差和速度

固定点条件只要求期望的 `ρδ=0`，因此在固定点处 `c` 不改变解。`c` 改变的是有限步目标和学习动态：

- 较小的 `c_bar` 衰减远处 `ρδ`，通常降低长轨迹目标方差；
- 较小的 `c_bar` 让远端奖励沿轨迹传播更慢，信用分配和 critic 收敛会变慢；
- 较大的 `c_bar` 加快远端奖励传播，但长乘积通常带来更高方差；
- `c_bar>0` 保证无限 horizon 期望递推仍可到达同一固定点；`c_bar=0` 只保留一步 `ρδ`，固定点需靠重复迭代达到。

`c_trace_speed_and_fixed_point` 用确定性 6 步链展示：`c=1` 一个 batch 到达固定点，`c=0.25` 需要 6 个 batch；二者固定点相同。

## 策略梯度与 stop-gradient

`vtrace_lab/learner.py` 的 PyTorch 批量实现满足：

- `behavior_probs` 是常数，不接收梯度；
- target 中使用的 `target_probs` 被 detach，避免 target/IS 比率把梯度回传给 learner；
- critic 输入 `values` 在构造 target 时 detach；`V_s^{target}` 整体 detach；
- policy loss 使用当前 learner 的 log-prob，但 advantage 本身 detach：

\[
L_{\text{actor}}=-\log\pi_\theta(a_t\mid s_t)\,A_t.
\]

- critic loss 为 mask 后的 MSE；padding 不参与 actor 或 critic loss。

测试显式检查行为概率无梯度、actor log-prob 有梯度、target 不向 critic 值回传，以及 padding loss 为零。

## 精确值与实验分离

精确部分使用线性方程，不靠采样估计：

- `solve_state_values`：求任意给定策略的 Bellman 固定点；
- `solve_vtrace_fixed`：求截断 V-trace 的期望固定点；
- `expected_vtrace_target`：求有限 horizon V-trace target 的精确期望，区分真实终止与截断 bootstrap。

实验部分单独报告：

- **固定点偏差**：`V_{π_ρ}-V_π`，增加样本不会消失；
- **有限样本误差**：bootstrap 使用精确 `V_{π_ρ}` 时，多随机种子的 RMSE/bias/std；
- **实际样本数**：真实终止后的 padding 不计入 `actual_transitions_mean`。

20 个种子的默认结果在 `results/experiment.json`。H=1 和 H=3 的有限样本 RMSE 都随实际样本数下降；同时 `ρ_bar=2` 的固定点偏差保持 `-0.2295`（滞后链 state 0），不会因增加样本消失。

## 明确的“降方差但引入目标偏差”反例

`bias_variance_bandit` 是一步终止 MDP：

- `μ=(0.2,0.8)`，`π=(0.8,0.2)`；
- 动作 0 奖励 `0.8`，动作 1 奖励 `0`；
- `Vπ=0.64`，`Vμ=0.16`。

从 `V=0` 的一步 target 看：

- `ρ_bar=∞`：方差 `1.6384`，期望 `0.64`，目标是原始 `π`；
- `ρ_bar=2`：方差 `0.4096`，但固定点变成 `V_{π_ρ}=0.5333...`，相对 `Vπ` 的固定点偏差为 `-0.1067`。

这就是截断带来的标准偏差–方差权衡：降低有限样本波动，但优化/评估对象不再是原始目标策略。

## 主要文件

- `vtrace_lab/mdp.py`：有限 MDP 与策略校验。
- `vtrace_lab/exact.py`：精确 Bellman/V-trace 线性方程、截断策略、期望目标。
- `vtrace_lab/vtrace.py`：逐轨迹参考版与 padding 批量 NumPy 版。
- `vtrace_lab/learner.py`：PyTorch 批量 target、stop-gradient、actor/critic loss。
- `vtrace_lab/sampler.py`：可控策略滞后采样器和版本/概率记录。
- `vtrace_lab/experiment.py`：复现实验入口。
- `tests/`：固定点、支持集、终止/截断、padding、策略滞后和梯度测试。
