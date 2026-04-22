实验记录：

4.15 
no activation function for q22
prox-cf, sra/oracle-nocf 
2000 linera+nn S1+S2
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
```结果：S1收敛很差(nn>linear); S2很差(linear>nn)

4.18
加softplus试试 recp 20
加leakyrelu recp25
```结果：加了sp，效果好于lr（几乎没效果）

4.19
*修改grid search的范围
***修改dgp的随机种子
跑 recp 28 5fold ————不是fold的问题

4.20
***修改q11的prepare tensor————逻辑修复; q22的loss:（q11*tt1）
***grid search + hejak estimator
实验：S1*2(linear+nn), S2*2(linear+nn) + S2(phi 2,3,4,0), all 30 recps


4.21 
*恢复grid search的范围
*修改dgp S2 Z2 var
*sra、oracle+ao/scl 控制


实验：只跑S2(baseline: S2-linear/nn-30)，需对比sp/lr的好坏（S2-nn-31/32），对比是否trim(no trim:S2-nn-33)， 对比fold数量的影响（S2-5fold-linear-20), 对比phitype(S2-nn-phi3/phi4-30)
结果：
1. 相比420：linear差不多，nn表现下降（方差增大）
2. 加入sp/lr的效果：均有改善，sp效果（4.35）明显好于lr
3. notrim效果：比普通nn好
4. phi3：效果更好（4.24）
5. 改成5fold：没有明显改善

4.22
在原setting基础上：nn+phi3+notrim+sp (tmux 3-3000,2-2000)
