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
*** 考虑kernel估计q22

