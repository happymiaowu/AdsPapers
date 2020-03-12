<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

### Practical Lessons from Predicting Clicks on Ads at Facebook

----
#### 论文地址
http://quinonero.net/Publications/predicting-clicks-facebook.pdf

#### 模型
GBDT+LR

#### 评价指标
1. Normalized Cross Entropy(NE)$$NE=\frac{-\frac{1}{N}\sum_{i=1}^{n}(\frac{1+y_i}{2}\log(p_i) + \frac{1-y_i}{2}\log(1-p_i))}{-(p*\log(p) + (1-p)\log(1-p))}$$

2. Calibration: $$r = \frac{\text{平均预测的CTR值}}{\text{历史CTR值}}$$

