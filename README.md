# Machine-Learning
This is my first ML project.

手写数字识别
（注：CNN）

一、准备工作

1.收集数据

    1）可以手动下载数据在http://yann.lecun.com/exdb/mnist/

    2)当然代码也会自己下载数据集

（注：国内小伙伴最好手动下载）

2.选择语言以及第三方库
    python

    TensorFlow

3.选择学习方法
    4层神经网络

4.IDE选择
    pycharm

二、实际操作
大致逻辑：三个函数即可

            add_layer（） 搭建神经网络的隐藏层和输出层

            build_nn()        把四层神经网络连接起来
            train_nn()         载入数据训练神经网络  



1.首先导入所需要的库，还有利用代码下载网络上的数据集


2.载入数据集，并且定义输入的数据格式，因为每张图片都是28*28像素的图片，所以要把他们转化为矩阵的格式就是784个单位，输出的格式呢是10个单位，以为0-9有十种，所一定一个1*10的矩阵输出。
3.构建隐藏层和输出层
4.连接神经网络
5.训练神经网络

先载入数据


4
在机器学习中的训练中往往会造成损失（loss），为了我们的模型更加准确我们要降低他们的损失，所以这里我们选择梯度下降（GradientDescent）的方法来降低我们的损失。


5
在我们之前已经定义了x，y等变量，但是这个时候他还没有值，TF中有sess这样的指针，可以只想当前计算的元素来赋予值，这样就把我们真个程序联动了起来


6
通过两个循环进行神经网络的训练，并且计算出每次训练后的错误率


7
计算最后的准确率，并且打印出来


8
最后的结果展示




9

在这之余还学习了用TensorBoard的可视化和利用web可视化，但是使用并不熟练，博文会在后期补充。
