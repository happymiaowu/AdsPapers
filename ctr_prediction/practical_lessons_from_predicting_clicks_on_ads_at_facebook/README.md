### Practical Lessons from Predicting Clicks on Ads at Facebook

----
#### 论文地址
http://quinonero.net/Publications/predicting-clicks-facebook.pdf

#### 模型
GBDT+LR

#### 评价指标
1. Normalized Cross Entropy(NE)
$$NE=\frac{-\frac{1}{N}\sum_{i=1}^{n}(\frac{1+y_i}{2}\log(p_i) + \frac{1-y_i}{2}\log(1-p_i))}{-(p*\log(p) + (1-p)\log(1-p))}$$
其中$p$为训练数据中的ctr值。除以训练数据中的ctr的熵的主要原因是排除训练数据的CTR值的影响，该值越小越好

2. Calibration:
$$r = \frac{\text{平均预测的CTR值}}{\text{历史CTR值}}$$
该值越接近1，结果越好

关于文章中为啥不采用AUC做评价指标：因为AUC只对排序敏感，不对具体的值敏感，但是点击率预估需要对具体值敏感。

#### 模型部分

GBDT部分是对特征做变换。对连续型参数转换有几种方法（文中采用第三种方法）：
1. (simple trick)对连续的参数进行分段，将每段作为一个分类特征
2. (simple but effective)进行特征组合，离散变量采用笛卡尔积的方式，连续变量采用joint binning方式，例如使用kd-tree
3. (powerful and very convenient way)GBDT，训练方式为: L2-TreeBoost algorithm

对于该模型文中举了如下例子：
![image](https://github.com/happymiaowu/AdsPapers/blob/master/ctr_prediction/practical_lessons_from_predicting_clicks_on_ads_at_facebook/pic/1.jpg)
在该图中，一共有两棵树，第一棵树有三个叶子节点、第二棵树有两个，如果样本x落入第一棵树的第二个叶子节点以及第二棵树的第一个叶子节点，那么该样本通过GBDT得到的向量为[0,1,0,1,0]，并将该向量作为LR的输入进行训练。

#### 训练部分
1. 由于随着时间推移，模型的预测效果越来越差，因此需要考虑训练频率。作者使用单核CPU用GBDT训练百亿级别数据，分成几百棵树，需要超过24小时，因此，作者建议树模型可以一天或者几天训练一次，线性分类器可以通过流式系统实时训练，算法是SGD。

2. 作者讨论了不同的学习率对LR＋SGD对结果的影响，发现最优的学习率为$$\eta_{t,i}=\frac{\alpha}{\beta+\sqrt{\sum_{j=1}^{t}\nabla_{j,i}^2}}$$
对于全局学习率 $\eta_{t,i}=\frac{\alpha}{\sqrt{t}}$ 效果不好的原因可能是对于样本量较少的特征，它们的收敛速度过快，难以收敛到最优值，而对于基于每个特征不同的学习率 $\eta_{t,i}=\frac{\alpha}{n_{t,i}}$ 效果不好的原因是虽然对不同的特征作区分，但是对于所有特征来说也存在收敛速度过快的问题。

#### 工程化部分
1. 用流式系统进行训练时，因为用户不会主动发请求告诉服务器“老纸就是不点这个广告”，因此对于每一个流量需要有一个时间窗口用于接收该流量的点击事件，如果超过这个时间窗口还没有收到点击事件，则主动记为“未点击”。因此，得到的数据集和真实的情况相比会有一定的偏差，偏差程度和模型的实时程度需要有一个trade off。从工程上实现方式是一个HashQueue，即Queue+HashMap。当有展示信息传入时，进Queue；当有点击信息传入时，进HashMap。当时间窗口结束时，流量出队，从HashMap中寻找该流量是否被点击，如果找不到该流量信息则表示未被点击。

2. 由于是在线系统，需要防止因为系统原因导致流量被污染。例如，当点击监控这条路因为系统原因阻塞时，意味着新来的的流量都没有点击，进入线上系统之后，将会导致模型预测的CTR值越来越低，引发广告展现量低的情况。因此，需要有监控机制和应急反应机制来解决这个问题。例如，当发现当前接受的数据分布突然异常时，立刻自动停止在线学习的行为

#### 结论：
1. NE下降基本来自于前500棵树，第1000棵树到第2000棵树只降低了不到0.1%
2. TOP10的feature占了一半的importance，最后300个feature贡献了不到1%的importance
3. 历史数据比上下文数据更重要，importance排TOP10的数据都是历史数据
4. 但是上下文数据对冷启动(新用户、新广告)更有帮助
5. 上下文数据比历史数据对时间更敏感
6. 拿所有数据进行训练和抽取10%的数据训练优化的量在1%(他们的数据集太大)
7. 如果减少负采样比例，相当于调整了历史CTR，因此预测的结果需要再调整回原来情况。调整公式为(why?)$$q=\frac{p}{p+\frac{1-p}{w}}$$

#### 思考
1. 调整公式推导:
> 假设通过负采样比例$w$采样后的训练集中的点击率为$p$, 正样本为$m$，负样本为$n$，那么有：
>
> 采样后的训练集点击率为$p=\frac{m}{n+m}$，采样前的训练集点击率为$q=\frac{mw}{n+mw}$
>
> 因此有$\frac{p}{1-p}=\frac{m}{n}$, $\frac{q}{1-q}=\frac{mw}{n}$
>
> 将$w$除至左边可得$\frac{p}{1-p}=\frac{q}{(1-q)w}$，整理得$q=\frac{pr}{1-p+pr}$，即$q=\frac{p}{p+\frac{1-p}{w}}$
