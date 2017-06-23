
- 添加 generate_neighbourhood_features ，LB Score 0.31731
- 去掉 generate_neighbourhood_features 添加经纬度信息，LB Score 0.31693
- 两个特征放一起是模型变差！

可以看出 generate_neighbourhood_features 的特征 和 经纬度特征 存在可能某种相关，
但两个特征单独使用都使模型提高！可以考虑经纬度和其他特征进行某种融合。
