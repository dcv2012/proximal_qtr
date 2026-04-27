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