# 딥러닝 용어사전

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

Adagrad는 적응형 학습률 변화기법입니다. 기울기의 제곱을 시간 변화에 따라 계속 유지하도록 하는 것이 특징이다. 기본 SGD를 대신항 사용할 수 있으며, 희소 데이터 (sparse data)에 사용하기 좋다. 희소 데이터에 높은 학습률을 할당하면 불규칙적으로 파라미터가 업데이트 되기 때문이다. (불규칙적인 데이터에 대해서도 방향성을 유지한다)

* [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf)
* [Stanford CS231n: Optimization Algorithms](http://cs231n.github.io/neural-networks-3/)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)


**ADAM**

Adam은 적응형 학습률 변화기법입니다. rmsprop과 유사하지만, 업데이트는 현재 기울기의 1차, 2차 모멘트, 편향(bias)를 이용해 즉시 추정됩니다.

* [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)


**AFFINE LAYER**

뉴럴 네트워크에서 완전 연결된 레이어 (fully-connected layer)입니다. Affine은 이전 레이어의 각 뉴런들이 현재 레이어의 각 뉴런들에 연결되어 있음을 뜻합니다. 여러모로, 이것은 뉴럴 네트워크의 "표준" 레이어입니다. Affine layer는 종종 Convolutional Neural Networks나 Recurrent Neural Networks의 가장 상위 출력에서 최종 예측을 하기 이전에 추가됩니다. Affine layer는 일반적으로 y = f(Wx + b)의 형태로 나타내며, x는 입력 레이어, W는 파라미터, b는 편향 (bias), f는 비선형 활성화 함수입니다. 


**ATTENTION MECHANISM**

Attention 메커니즘은 이미지에서 특정 부분에 집중하는 사람의 시각적 주목 능력에 영감을 받았습니다. Attention 메커니즘은 자연어 처리와 이미지 인식 아키텍쳐에서 네트워크가 예측하기 위해서 주목해서 학습해야 하는 것에 대해 표현 가능합니다. 

* [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)


**ALEXNET**

Alexnet은 이미지넷 대회 (ILSVRC 2012)중 이미지 인식에서 가장 큰 성능차이로 우승했던, Convolutional Neural Networks (CNN) 구조이며, CNN의 관심을 부활시킨 장본인입니다. 5개의 convolutional 레이어와 그 뒤 일부분의 max-pooling 레이어로 구성되어 있으며, 3개의 완전 연결된 (fully-connected) 레이어와 1000개의 출력을 가진 softmax를 포함합니다. Alexnet은 [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)에 소개되었습니다.


**AUTOENCODER**

Autoencoder는 뉴럴 네트워크 모델 중 입력 자체를 예측하기 위한 목적을 가진 모델입니다. 네트워크 내부에 병목지역 ("bottleneck")을 포함하고 있는 것이 특징입니다. 병목지역부터 소개하자면, 네트워크로 하여금 입력에 대한 저차원 표현법을 학습하게끔 합니다. 효율적으로 입력에 대한 좋은 표현법으로 압축하는 것이죠. Autoencoder는 PCA나 다른 차원 축소 기법과 관련이 있습니다. 하지만 비선형적인 특성때문에 더 복잡한 연관성을 학습할 수 있습니다. [Denoising Autoencoders](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf), [Variational Autoencoders](http://arxiv.org/abs/1312.6114), [Sequence Autoencoders](http://arxiv.org/abs/1511.01432)와 같이 다양한 형태의 autoencoder 모델이 있습니다.

**AVERAGE-POOLING**

Average-Pooling은 이미지 인식을 위해 Convolutional Neural Networks에 사용되는 pooling 기법중 하나입니다. 특징이 표현된 패치 위를 윈도우가 순회하며 해당 윈도우의 모든 값의 평균을 취합니다. 이는 입력 표현을 저차원의 표현으로 압축하는 역할을 합니다.



B
--------------

**BACKPROPAGATION**

Backpropagation은 뉴럴 네트워크에서 (혹은, 일반적인 feedforward 그래프에서) 효율적으로 경사(gradient) 를 계산하기 위한 방법입니다. 네트워크 출력으로 부터 편미분 연쇄 법칙을 이용해 경사도를 계산하여 입력쪽으로 전달합니다. Backpropagation의 첫 사용은 1960년대의 Vapnik의 사례로 거슬러 올라갑니다. 하지만 [Learning representations by back-propagating errors](http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html)이 근원이라고 종종 인용되고 있습니다. 

* [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/)


**BACKPROPAGATION THROUGH TIME (BPTT)**

Backpropagation Through Time ([논문](http://deeplearning.cs.cmu.edu/pdfs/Werbos.backprop.pdf))은 Backpropagation 알고리즘이 Recurrent Neural Networks (RNNs) 에 적용된 것입니다. BPTT는 RNN에 적용되는 표준적인 backpropagation 알고리즘이라고 볼수 있습니다. RNN은 각 시점이 recurrent 레이어를 나타내며, 각 recurrent 레이어는 파라미터를 공유합니다. BPTT는 이름 그대로, RNN이 모든 시점에 통해 같은 파라미터를 공유하기 때문에, 한 시점의 에러는 모든 이전 시점에게 "시간을 거슬러 (through time)" 전달됩니다. 수백개의 입력으로 구성된 긴 sequences를 처리할 때, 계산 효율을 위해 truncated BPTT가 종종 사용됩니다. Truncated BPTT는 정해진 시점까지만 에러를 전달하고 멈춥니다. 

* [Backpropagation Through Time: What It Does and How to Do It](http://deeplearning.cs.cmu.edu/pdfs/Werbos.backprop.pdf)


**BATCH NORMALIZATION**

Batch Normalization은 레이어의 입력을 mini-batch로 정규화 하기 위한 기법입니다. 학습 속도를 높여주고, 높은 학습률을 사용 가능하게 하며, 정규화 (regularization) 하도록 합니다. Batch Normalization은 Convolutional Neural Networks (CNN)과 Feedforward Neural Networks (FNN)에 아주 효과적이라고 밝혀졌습니다. 하지만 Recurrent Neural Networks 에는 아직 성공적으로 적용되지 않았습니다. 

* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)
* [Batch Normalized Recurrent Neural Networks](http://arxiv.org/abs/1510.01378)


**BIDIRECTIONAL RNN**

Bidirectional Recurrent Neural Network은 두개의 서로 다른 방향을 가진 RNN을 포함하는 뉴럴 네트워크를 의미합니다. 순방향 (forward) RNN은 입력 sequence를 처음부터 끝까지 읽고, 역방향 (backward) RNN은 끝에서 부터 처음의 방향으로 읽습니다. 두 RNN은 두가지 방향 벡터로 볼수 있으며, 입력 레이어 상단에 두 RNN을 쌓고, 그 위에 출력 레이어로 묶게 됩니다. Bidirectional RNN은 자연어 처리 문제에서 종종 사용됩니다. 특정 단어의 앞, 뒤 단어의 의미를 통해 현재 단어를 예측하는 문제에 적용됩니다.

* [Bidirectional Recurrent Neural Networks](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)



C
--------------

**CAFFE**

[Caffe](http://caffe.berkeleyvision.org/)는 Berkeley Vision and Learning Center에서 개발된 딥러닝 프레임워크입니다. Caffe는 특히 비전관련 영역과 CNN 모델을 처리하는데 특화되어 있습니다. 


**CATEGORICAL CROSS-ENTROPY LOSS**


**CHANNEL**

**CONVOLUTIONAL NEURAL NETWORK (CNN, CONVNET)**



D
--------------

**DEEP BELIEF NETWORK (DBN)**

**DEEP DREAM**

**DROPOUT**


E
--------------

**EMBEDDING**

**EXPLODING GRADIENT PROBLEM**


F
--------------

**FINE-TUNING**


G
--------------

**GRADIENT CLIPPING**

**GLOVE**

**GOOGLELENET**

**GRU**


H
--------------

**HIGHWAY LAYER**



I
--------------
**ICML**

**ILSVRC**

**INCEPTION MODULE**


K
--------------
**KERAS**


L
--------------
**LSTM**


M
--------------
**MAX-POOLING**

**MNIST**

**MOMENTUM**

**MULTILAYER PERCEPTRON (MLP)**


N
--------------

**NEURAL MACHINE TRANSLATION (NMT)**

**NEURAL TURING MACHINE (NTM)**

**NONLINEARITY**

**NOISE-CONTRASTIVE ESTIMATION (NCE)**


P
--------------
**POOLING**


R
--------------

**RESTRICTED BOLTZMANN MACHINE (RBM)**

**RECURRENT NEURAL NETWORK (RNN)**

**RECURSIVE NEURAL NETWORK**

**RELU**

**RESNET**

**RMSPROP**

S
--------------

**SEQ2SEQ**

**SGD**

**SOFTMAX**


T
--------------

**TENSORFLOW**

**THEANO**


V
--------------

**VANISHING GRADIENT PROBLEM**

**VGG**


W
--------------

**WORD2VEC**





## References

* [Deep Learning Glossary @ wildml.com](http://www.wildml.com/deep-learning-glossary/)