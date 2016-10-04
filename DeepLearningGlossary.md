# 딥러닝 튜토리얼

A
--------------

**ACTIVATION FUNCTION**

뉴럴 네트워크가 복잡한 결정 경계 (decision boundaries)를 학습하기 위해서, 특정 레이어에 비선형 활성화 함수를 적용합니다. 일반적으로 많이 사용되는 함수는 [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [tanh](http://mathworld.wolfram.com/HyperbolicTangent.html), [ReLU (Rectified Linear Unit)](http://cs231n.github.io/neural-networks-1/)이며, 다양한 변형 기법도 사용됩니다. 


**ADADELTA**

Adadelta는 경사강하 기반 학습 기법으로, 학습률을 시간이 변화함에 따라 조정합니다. 하이퍼파라미터에 민감하며, 너무 빠르게 학습률이 감소하는 [Adagrad]보다 향상시키기 위해 제안되었습니다. Adadelta는 [rmsprop] 기법과 유사하며 기본 [SGD]를 대신하여 사용할 수 있습니다.

* [ADADELTA: An Adaptive Learning Rate Method](http://arxiv.org/abs/1212.5701)
* [Stanford CS231n: Optimization Algorithms](http://cs231n.github.io/neural-networks-3/)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)


**ADAGRAD**

Adagrad는 적응형 학습률 변화기법입니다. 