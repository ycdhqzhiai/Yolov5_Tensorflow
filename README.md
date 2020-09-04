# Yolov5_Tensorflow
**Fork from https://github.com/avBuffer/Yolov5_tf**</br>

# 训练

简单测试了下，使用[yolov3](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3) 面的yymnist数据集

```shell
$ git clone https://github.com/YunYang1994/yymnist.git
$ python yymnist/make_data.py --images_num 1000 --images_path ./data/dataset/train --labels_txt ./data/dataset/yymnist_train.txt
$ python yymnist/make_data.py --images_num 200  --images_path ./data/dataset/test  --labels_txt ./data/dataset/yymnist_test.txt
```

针对avBuffer仓库的yolov5存在的问题做了部分修改，目前在yolov5s上测试通过，去除了Focus模块的slice op(为了移植，我在原版上面测试过，好像并没有掉点)
```python
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        ##[[-1, 1, Focus, [64, 3]],  # 0-P1/2
        #self.conv = Conv(c1 * 4, c2, k, 1)
        self.conv = Conv(c1, c2, k, 2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        #re = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        re = self.conv(x)
        return re
```
**测试时候训练开始正常收敛，根据选择loss不同，训练loss在第5-7个epoch会出现NA，修改ing**

## 9.4更新
* 1.修改了csp2结构，去除了部分bn
