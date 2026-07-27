# 项目介绍和规划
本项目是蒸馏项目
现有方法大多分别处理“采样步数”和“网络深度”，而且现有蒸馏方法通常只匹配最终速度或最终动作，却没有蒸馏Teacher如何沿着网络深度和去噪时间逐步形成动作决策。

## 1. 增加 DiT 深度节点输出接口
### 修改文件：
action_model/models.py
action_model/action_model.py
新增或补充对应单元测试
### 实现内容：
保持现有 forward() 默认行为不变。
增加可选参数，记录每个 Transformer block 后的输出。
使用共享 final_layer 将中间特征投影为 eps_l。
返回最终噪声预测和各深度节点的噪声预测，统一形状为 [L,B,T,C]。
### 风险点：
对每层都调用 final_layer 会增加显存和计算量。
中间特征分布未必适合直接使用最终输出层。
改变返回类型可能破坏训练和推理调用方。
### 验收方法：
默认调用结果与修改前数值一致。
DiT-B 返回 12 个深度节点，DiT-S 返回 6 个。
每个节点形状为 [B,16,7]。
完成 forward、backward 和 CFG shape 测试。

## 2. 将中间噪声转换为统一动作估计
### 修改文件：
action_model/gaussian_diffusion.py
action_model/respace.py，仅在时间步映射确有需要时修改
distillation/runners.py
distillation/test.py
### 实现内容：
根据当前 x_t、扩散系数和 eps_l 计算每层的 x0_l。
Teacher 在每个 DDIM step 中记录所有深度节点。
按实际执行顺序展开为“时间 × 深度”路径。
Student 使用同样机制记录少步、少层路径。
### 风险点：
DDIM 的重映射时间步与原始扩散时间步容易混淆。
一个 step 内各 block 共享同一个 x_t，中间节点表示的是逐层改善的 x0 估计，不是实际经历的 DDIM 状态。
Teacher CFG 必须先正确合成条件和无条件预测，再转换为 x0。
### 验收方法：
最后一层得到的 x0_l 与现有 DDIM pred_xstart 数值对齐。
Teacher 20×12 得到 240 个节点，Student 4×6 得到24个节点。
关闭深度记录时，原轨迹蒸馏结果保持不变。
检查所有节点无 NaN/Inf。

## 3. 实现细化进度压缩与路径损失
### 修改文件：
新增 distillation/path.py
distillation/loss.py
conf/distillation.py
distillation/test.py
### 实现内容：
根据相邻 x0 动作估计的变化量计算节点距离。
归一化累计距离，得到 [0,1] 的动作细化进度。
将 Teacher 的约 240 个节点插值为与 Student 相同的24个锚点。
新增：
L_path  = Student 节点与 Teacher 锚点的状态对齐
L_macro = 相邻节点动作更新量对齐
增加开关和权重，使现有 L_traj baseline 仍可运行。
### 风险点：
初始阶段动作变化量可能极小，累计进度分母需要稳定处理。
夹爪维度与位移、旋转量纲不同，直接 L2 可能让某些维度主导距离。
插值后的锚点不一定对应真实 Teacher 层节点。
### 验收方法：
人工构造线性路径，验证进度和插值结果精确。
恒定路径不出现除零或 NaN。
锚点首尾与 Teacher 路径首尾一致。
L_path、L_macro 可独立反向传播到 Student，Teacher 无梯度。

## 4. 接入训练、评测和阶段性实验
### 修改文件：
distillation/train.py
scripts/distillation_cogact.py
scripts/eval_distillation.py
conf/distillation.py
train_eval.txt
必要时补充 DISTILLATION_PLAN.md
### 实现内容：
在训练循环接入深度路径记录和新损失。
日志记录节点数、各项损失、路径长度、梯度和显存。
离线评测增加最终 MSE、路径 MSE、宏更新 MSE。
设计逐级消融：
A: 当前 Ltask + Lfinal + Ltraj
B: A + Lpath
C: B + Lmacro
第一阶段暂不加入视觉语言贡献蒸馏，先验证深度路径是否有效。
### 风险点：
Teacher 240 节点全部保留可能显著增加显存。
训练速度下降会抵消实验迭代效率。
新路径损失尺度可能压过 Ltask 和 Lfinal。
### 验收方法：
batch_size=1 完成完整 forward/backward/checkpoint。
短跑若干 step，损失有限且 Student 梯度非零。
对比开启/关闭新方案的显存和单步耗时。
checkpoint 能由 SimplerEnv 正常加载，推理接口和 action shape 不变。

## 5. Teacher 与 GT 分工监督
### 可行性与目标：
该方案可行。Teacher 的优势是提供稠密、平滑的动作生成过程，GT 的优势是提供不受 Teacher 偏差影响的真实任务终点。因此将监督拆分为：

- Teacher：监督 Student 如何逐步形成动作，即时间轨迹、深度路径和宏更新。
- GT：监督 Student 最终生成的 action chunk，即 Final Action。

推荐采用严格分工版本。当前 L_traj 和 L_path 都包含最后节点；如果直接保留，Teacher 仍会通过末端轨迹间接约束 Final Action。因此严格分工时，需要从 Teacher 过程损失中屏蔽最后状态和最后一次宏更新。

### 推荐损失：
```text
L_final_gt = MSE(x0_student, action_gt)

L_traj_teacher = MSE(
    student_trajectory[:-1],
    teacher_trajectory[:-1]
)

L_path_teacher = MSE(
    student_depth_path[:-1],
    teacher_anchors[:-1]
)

L_macro_teacher = MSE(
    student_updates[:-1],
    teacher_anchor_updates[:-1]
)

L_total = lambda_task * L_task
        + lambda_final_gt * L_final_gt
        + lambda_traj_teacher * L_traj_teacher
        + lambda_path_teacher * L_path_teacher
        + lambda_macro_teacher * L_macro_teacher
```

其中 `L_task` 是基于 GT 动作的原始扩散噪声预测损失。建议第一版保留为稳定训练的基础正则项，并通过消融实验判断是否可以降低或移除。原来的 Teacher Final Loss：

```text
MSE(x0_student, x0_teacher)
```

在严格分工实验中关闭，仅保留为兼容 baseline 的配置开关。

### 修改文件：
- `distillation/loss.py`
- `distillation/path.py`
- `conf/distillation.py`
- `distillation/train.py`
- `distillation/train_utils.py`
- `scripts/eval_distillation.py`
- `tests/distillation/test_path.py`
- 新增或补充 GT/Teacher 分工损失测试

### 实现内容：
1. 新增 `L_final_gt = MSE(x0_student, actions_future)`，明确使用数据集 GT 监督最终 action chunk。
2. 保留现有 `L_final_teacher = MSE(x0_student, x0_teacher)`，但独立配置权重，严格分工时设为 0。
3. 为 `L_traj` 增加 `exclude_terminal=True`，只对齐 Teacher 和 Student 的非终端 DDIM 节点。
4. 为 `depth_path_losses()` 增加终端屏蔽选项：
   - `L_path` 排除最后一个 Student 节点和 Teacher 最终锚点；
   - `L_macro` 排除通向最终节点的最后一次更新。
5. 配置项建议调整为：

```text
lambda_task
lambda_final_gt
lambda_final_teacher
lambda_traj_teacher
lambda_path_teacher
lambda_macro_teacher
exclude_teacher_terminal = True
```

6. 日志同时记录：

```text
loss/final_gt
loss/final_teacher
loss/traj_teacher
loss/path_teacher
loss/macro_teacher
student_gt_action_mse
student_teacher_final_mse
teacher_gt_action_mse
```

### 风险点：
1. DDIM 最终样本直接对齐单条 GT 演示，可能削弱动作分布的多模态性；需要观察 rollout，而不能只看离线 MSE。
2. GT 动作可能包含遥操作噪声，Teacher 轨迹更平滑，二者在末端附近可能产生梯度冲突。
3. 屏蔽 Teacher 末端后，最后一次 Student 更新只受 GT 监督，可能出现末端跳变。
4. `L_task` 本身也是 GT 监督。如果论文强调完全分工，需要明确它是基础扩散训练项，或者通过实验将其权重设为 0。
5. 若 Student/Teacher 节点数很少，排除终端可能使过程监督节点不足；单节点路径应直接退化为仅 GT Final Action 监督。

### 风险控制：
- 记录 Student 最后两节点的更新幅度，监控是否出现末端跳变。
- 分别记录 `student_gt_action_mse`、`student_teacher_final_mse` 和 `teacher_gt_action_mse`，判断 GT 与 Teacher 是否存在系统偏差。
- 对 GT Final Loss 使用独立权重和可选 warmup，避免训练初期压过路径监督。
- 如严格分工不稳定，可退回软分工：Teacher 路径包含末端，但降低 Teacher Final 权重。

### 消融实验：
```text
A. 原方案：Teacher Final + Teacher Trajectory/Path
B. 软分工：GT Final + 完整 Teacher Trajectory/Path
C. 严格分工：GT Final + Teacher 非终端 Trajectory/Path（推荐）
D. 严格分工但关闭 L_task：验证是否真正只由 Teacher 过程 + GT 终点驱动
```

统一比较：

- rollout success rate
- student_gt_action_mse
- student_teacher_final_mse
- 末端动作跳变量
- 动作抖动程度
- policy latency 和 GPU memory

### 验收方法：
1. 单元测试确认 GT Final Loss 只依赖 `actions_future`，不依赖 `x0_teacher`。
2. 修改 Teacher 最终节点时，严格分工下 `L_traj_teacher` 和 `L_path_teacher` 数值保持不变。
3. `L_final_gt.backward()` 能向 Student 传播梯度，Teacher 始终无梯度。
4. 单节点 Student 路径不会产生空张量 MSE 或 NaN。
5. `batch_size=1` 完成 forward、loss、backward 和 checkpoint。
6. 至少完成 A/B/C 三组消融，并以 SimplerEnv rollout success rate 作为最终判断依据。

## 另外
建议先完成以上五步，再单独进入第二阶段实现“视觉语言控制贡献蒸馏”。这样可以把深度路径机制与 CFG 双分支机制的风险分开验证。
