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
***（重要！！！）修改q11的prepare tensor————逻辑修复; 
q22的loss:（q11*tt1）
***grid search + hejak estimator
！！丢弃420之前的所有实验结果

实验：S1*2(linear+nn), S2*2(linear+nn) + S2(phi 2,3,4,0), all 30 recps



4.21 
*恢复grid search的范围
*修改dgp S2 Z2 var
*sra、oracle+ao/scl 控制

实验：
1. 只跑S2(baseline: S2-linear/nn-30)，
2. 对比sp/lr的好坏（S2-nn-31/32），
3. 对比是否trim(no trim:S2-nn-33)，  
4. 对比phitype(S2-nn-phi3/phi4-30)
5. 对比fold数量的影响（S2-5fold-linear-20）

结果：
1. 相比420：linear差不多，nn表现下降（方差增大）
2. 加入sp/lr的效果：均有改善，sp效果（4.35）明显好于lr
3. no_trim效果：比普通nn好
4. phi3效果更好（4.24）, phi4更差
5. 改成5fold：没有明显改善



4.22
在原setting基础上：nn+phi3+notrim+sp (S2-3000-35,S2-2000-35)
结果：4.30；4.41

**修改dgp S2：增加Y0,Y1的var，减小W1的var
*增加q估计中损失关于q的显示l2-正则（lambda_reg）
**删掉trim

实验:
1. baseline: S2-linear/nn-30-phi1; S2-linear/nn-30-phi3     4组 
2. 对比q22loss：q11（原）与q11*tt1 —— S2-nn-31-phi1     1组
3. 加sp/lr：S2-nn-phi1-32/33 2组
4. 删掉est中的截断: S2-nn-phi1-34

结果：
1. baseline：S2-linear-phi1 (4.098), S2-nn-phi3(4.35)>nn-phi3
2. 原loss：更差（3.729）
3. S2-nn-phi1-sp：4.14，lr：4.00（较差）
4. 删掉est中的截断：4.2505
