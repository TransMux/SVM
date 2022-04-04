<h2 align="center">✨手搓SVM</h1>

<div align="center"><a href="README-en.md">--> Readme.md in English <--</a></div>

来自机器学习第四次实验~

实现了统一接口，注释比较多，新手友好~

欢迎issue或者pr，如果帮到了你也欢迎点个star~

### 目录结构

```tree
│   .gitignore
│   Circle.csv # 非线性可分的数据集
│   Last-Circle.csv.png # 非线性可分数据集实验效果
│   Last-linear_hard_margin.csv.png # 线性可分数据集实验效果
│   Last.png # 最后一次实验效果
│   linear_hard_margin.csv # 线性可分数据集
│   main.py # 入口程序
│   myplot.png # 某一次效果比较好的
│   README-en.md
│   README.md
└───model # 模型目录
    │   SVM.py # SVM本体
    └───algorithm # 算法
        │   kernels.py # 核函数
        │   smo.py # SMO 算法
        └───utils.py # 可视化工具
```

### 效果

![](Last-linear_hard_margin.csv.png)

![](myplot.png)

### License

GPL
