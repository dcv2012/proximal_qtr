实验记录：
4.14
*之前的方法逻辑问题，已丢弃前面的数据和记录

4.15 
no activation function for q22
prox-cf, sra/oracle-nocf 
2000 linera+nn s1
reps: 30
4 session

4.16
softplus, reps 20

4.17
对比方法考虑：SRA和ORACLE 是否应该除了nuisance的估计其余框架保持一致，还是直接对比XIA的方法(二分搜索、hingeloss)？
***改了cross-fitting的实现
***增强nan的判断稳健性
*增加内部截断
*--增加截断5/95
*grid search收敛阈值与二分法一样
实验：跑了4组 30reps
```结果：S1收敛很差(nn>linear); s1很差(linear>nn)

4.18
加softplus试试 recp 20
加leakyrelu recp25
```结果：加了sp，效果好于lr（几乎没效果）

4.19
*修改grid search的范围
***修改dgp的随机种子
跑 recp 28 5fold ————不是fold的问题

4.20
***（重要！！！）修改q11的prepare tensor————逻辑修复; 
q22的loss:（q11*tt1）
***grid search + hejak estimator
！！丢弃420之前的所有实验结果

实验 s1*2(linear+nn) + s1(phi 2,3,4,0), all 30 recps



4.21 
*恢复grid search的范围
*修改dgp s1 Z2 var
*sra、oracle+ao/scl 控制

实验：
1. 只跑s1(baseline: s1-linear/nn-30)，
2. 对比sp/lr的好坏（s1-nn-31/32），
3. 对比是否trim(no trim:s1-nn-33)，  
4. 对比phitype(s1-nn-phi3/phi4-30)
5. 对比fold数量的影响（s1-5fold-linear-20）

结果：
1. 相比420：linear差不多，nn表现下降（方差增大）
2. 加入sp/lr的效果：均有改善，sp效果（4.35）明显好于lr
3. no_trim效果：比普通nn好
4. phi3效果更好（4.24）, phi4更差
5. 改成5fold：没有明显改善



4.22
在原setting基础上：nn+phi3+notrim+sp (s1-3000-35,s1-2000-35)
结果：4.30；4.41

**修改dgp s1：增加Y0,Y1的var，减小W1的var
*增加q估计中损失关于q的显示l2-正则（lambda_reg）
**删掉trim

实验:
1. baseline: s1-linear/nn-30-phi1; s1-linear/nn-30-phi3     4组 
2. 对比q22loss：q11（原）与q11*tt1 —— s1-nn-31-phi1     1组
3. 加sp/lr：s1-nn-phi1-32/33 2组
4. 删掉est中的截断: s1-nn-phi1-34

结果：
1. baseline：s1-linear-phi1 (4.098), s1-nn-phi3(4.35)>nn-phi3
2. 原loss：更差（3.729）
3. s1-nn-phi1-sp：4.14，lr：4.00（较差）
4. 删掉est中的截断：4.2505



4.23
***「重要调整」DGP（增加异质项）: $Y_2 = -2.25 - 1.5A_1 - 2.25A_2 - 0.5A_1A_2 - 0.5Y_1 + 5U_1 + 4U_0 - 3A_2U_1$
原DGP: 6U1,6U0
*q22估计中：kernal2加入A1作为输入
**固定q22loss：统一为q11*tt1
*已经删掉est中的截断
**统一策略网络：kaiming初始化、relu激活

实验：
1. baseline（默认phi1,2000）：s1-linear/nn-30-phi1   
2. 对比phi3：s1-linear/nn-30-phi3   
3. 改变样本量、reps：s1-nn-1000-30-phi1, s1-nn-5000-30-phi1
4. 对比fold之后有没有变化：s1-nn-5fold-31
5. 对比U/V统计量的q22：s1-nn-32（V）

结果：
1. phi1--linear: prox/sra 4.5948/5.1514; nn: prox/sra 4.8107/5.1233
2. phi3--linear: prox/sra 4.9315/5.0941; nn: prox/sra 5.2214/5.1391 (win)
3. "1000" prox/sra 5.1111/5.1519; "5000" 
4. 出现了nan
5. V: prox/sra 5.3026/5.1167 (win)


4.24 好起来了！！！
**更正：之前的S2是新的S1，引入非线性regime S2
**修改dgp，引入U0的异质性：$$Y_2 \sim \mathcal{N}(-1.5 - 2.5 A_2 - 1.5 A_1 - 0.5 A_1 A_2 + 5.0 U_1 + 4.0 U_0 - 4.0 A_1 U_0 - 3.0 A_2 U_1, \, 0.2^2)$$
*恢复est中的截断（因为昨天4出现了nan）
*根据昨天结论：phi3>phi1, V>U

*** 效果最好：nn+phi3+Vstat
实验：
--S1--
1. baseline：S1-linear/nn+phi3+Vstat
2. 500，2000，5000对比: S1-nn-500-phi3
3. 再跑一个昨天的试一下：S1-nn-32-phi1

--S2--
1. baseline: S2-linear/nn-30-phi1/phi3  4组
2. 500,5000: S2-nn-500-phi3

--4.24 S1（Y2中删掉Y1）--
500--2000 5.22--5.3
1. 原S1-nn-phi3-V: 500,2000  29reps
2. kernal2 加入A1后：500,2000 28reps
3. 删去额外惩罚项：S1-nn-phi3-2000， 27reps

结果：新的S1、S2都不work
29>28>27 -- S1-nn-phi3-V


4.25
**修改DGP S1,S2

实验计划
1. 收敛性：1000，2000，5000
2. 对比phi
3. 对比model
4. 对比

实验
--S1-- no_cf
1. baseline：S1-nn-phi1/phi3-2000-30
2. S1-nn-phi3-1000/5000-30
3. S1-linear-phi1/phi3-30

--S2-- no_cf
1. baseline：S2-nn-phi1/phi3-2000-30
2. S2-nn-phi3-500/5000-30

不太好 -- 原因：策略学习也看不到U，所以不适合加入过高的Y*U项

4.26
**修改DGP S1,S2
实验:
base:4.24 dgp
S1-NN-PHI3-2000-U/V     31/32
结果：U<V

--S1-- no_cf
1. S1-nn-phi3-1000/2000/5000-30
2. S1-linear-phi3-2000-30

--S2-- no_cf
1. S2-nn-phi3-1000/2000/5000-30
2. S2-linear-phi3-2000-30

4.27
***q22估计中：kernel2加入A1作为输入（按 method.tex 修正）
***q22输出层加入 `C*tanh(.)` 有界化，只作用于q22；默认先固定 `C=5`
*保留q22可为负
*坏seed小规模测试 `C=3/4/5/6/8/10`：`C=3/4` 偏饱和，`C>=6` 方差和负值比例回升，`C=5` 最平衡

实验计划：
--S1-- comparative_analysis_427, no_cf
1. baseline：S1-nn-phi3-500/2000/5000-30
2. 对比model：S1-linear-phi3-2000-30
3. 对比phi：S1-nn-phi1-2000-30；S1-linear-phi1-2000-30
统一设置：tau=0.5, mmr_loss=V_statistic, q22_output_bound=5, comparative analysis, `--no_cf`

4.28
***修复：q22 的 `q22_output_bound` 传递到最终重训（避免默认回落 C=5）
***修复：inner grid 改为右侧可行根优先（避免低 q 根）
***修复：AO 增加 `min_alt_iters=2`
*删除本轮诊断临时代码，仅保留必要核心修复

实验计划（第1步）：
--S1-- comparative_analysis_428, no_cf
1. baseline：S1-nn-phi3-500/2000/5000-30
2. 对比model：S1-linear-phi3-2000-30
3. 对比phi：S1-nn-phi1-2000-30；S1-linear-phi1-2000-30
统一设置：tau=0.5, mmr_loss=V_statistic, q22_output_bound=5, `--no_cf`

4.29
*全局 `q22_output_bound` 默认值更新为 `5.5`
*结果目录切换为 `comparative_analysis_429`

实验计划（执行 exp.md 第1项）：
--S1-- comparative_analysis_429, no_cf
1. baseline：S1-nn-phi3-500/2000/5000-30（C=5.5）
2. 对比model：S1-linear-phi3-2000-30（C=5.5）
3. 对比phi：S1-nn-phi1-2000-30；S1-linear-phi1-2000-30（C=5.5）
4. 额外C对比（n=2000, phi3, nn）：C=3/4/6（其余保持 C=5.5）
统一设置：tau=0.5, mmr_loss=V_statistic, `--no_cf`

结果（最终）：
1. phi3-nn 不同 n：n=500/2000/5000 的 Prox 均值分别为 5.0533/5.2089/4.9082；SRA 为 5.1046/5.1071/5.0977。仅 n=2000 下 Prox 均值超过 SRA。
2. n=2000 下 model/phi：phi3-linear 的 Prox 均值最高（5.3670），phi1-nn 为 5.2676，phi3-nn 为 5.2089；但 linear-phi1 出现一次极端低值（-2.5808）。
3. n=5000 仍存在不稳定：Prox 标准差 1.2993，最差值 -0.0644，低于 4.5 的坏轮次 2 次。
4. 额外 C=3/4/6 对比结果未在独立目录中保留，无法可靠复原；本日正式结论仅基于 C=5.5 主实验。
5. 总体：4.29 比 4.30 稳定，但相较 4.28 未整体改善；Prox 尚不能稳定战胜 SRA。

4.30
**step1: 删去lambda_reg 额外正则化项
**step2: 改回argmin abs; 不使用hajek归一化
**estimate: 增加flip rate约束（f1、f2 < 5%），优化flip rate退出条件%0.01
**对SRA/oracle：删去hajek归一化
*根据429实验结果，统一设定C=3，默认baseline phi=1

实验：
额外：1.对C=4，5，6，跑n5000-phi1-nn-30reps
其他按照原有的实验计划来
*为避免 4.29 未完成任务污染，新增副本脚本：`Main/run_comparative_analysis_430.py`
*4.30 结果主目录：`comparative_analysis_430`
*额外 C 对比目录：`comparative_analysis_430_C4/C5/C6`


5.2 
*改回reg、hajek形式
*对比C=2，3，4，5
*对比输出层激活函数的影响（分别切换注释）
*eps、delta--1e-5

实验：
1. baseline：nn-phi3-500/2000/5000-C4
2. 对比C：nn-phi3-5000-C3/5/6
3. 对比输出层激活函数：手动修改切换 MMR_model 中输出层激活函数分别为relu、C*sigmoid情况进行测试，使用参数nn-phi3-5000-C4-reps31、32

**5.2 执行记录（2026-05-02，按 exp.md 读取本段后启动）**

- 结果写入：`Main/results/comparative_analysis_502/`（`run_comparative_analysis.py` 中 `RESULTS_DIRNAME`）。
- **解释器**：tmux 下须使用 `/home/ctl/.conda/envs/DEEPMMR/bin/python`（默认 `python` 可能无 `torch`，首次已全部用上述路径重启）。
- **入口脚本**：默认执行「MC → 汇总 → 箱线图」；若仅读取已有 CSV 做图表，可加 `--analyze_only`。原始 CSV/boxplot 文件名含 `_C{C}_`，与 `--q22_output_bound` 一致。
- **q22 输出激活消融**：不设环境变量为默认 `tanh`；设置 `MMR_Q22_OUTPUT_ACT=relu` 或 `MMR_Q22_OUTPUT_ACT=sigmoid`（等价于 `C*sigmoid`，有界至 `[0,C]`）。
- **tmux 会话名**：`ca502_n500_C4`、`ca502_n2000_C4`、`ca502_n5000_C4`、`ca502_n5000_C3`、`ca502_n5000_C5`、`ca502_n5000_C6`、`ca502_act_relu31`（`MMR_Q22_OUTPUT_ACT=relu`、`mc_reps=31`）、`ca502_act_sig32`（`sigmoid`、32）。
- **日志**：`Main/results/comparative_analysis_502/logs/<会话名>.log`。若长时间为 0 字节，为管道下 Python 块缓冲所致；如需实时可读可改用 `python -u` 或包一层 `stdbuf -oL` 后重启对应会话。
- **统一**：`--dgp S1 --no_cf --phi_type 3 --model_type nn --tau 0.5`（默认值未写的即脚本默认）。