# ������ ������

A
--------------

**ACTIVATION FUNCTION**

���� ��Ʈ��ũ�� ������ ���� ��� (decision boundaries)�� �н��ϱ� ���ؼ�, Ư�� ���̾ ���� Ȱ��ȭ �Լ��� �����մϴ�. �Ϲ������� ���� ���Ǵ� �Լ��� [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [tanh](http://mathworld.wolfram.com/HyperbolicTangent.html), [ReLU (Rectified Linear Unit)](http://cs231n.github.io/neural-networks-1/)�̸�, �پ��� ���� ����� ���˴ϴ�. 


**ADADELTA**

Adadelta�� ��簭�� ��� �н� �������, �н����� �ð��� ��ȭ�Կ� ���� �����մϴ�. �������Ķ���Ϳ� �ΰ��ϸ�, �ʹ� ������ �н����� �����ϴ� [Adagrad]���� ����Ű�� ���� ���ȵǾ����ϴ�. Adadelta�� [rmsprop] ����� �����ϸ� �⺻ [SGD]�� ����Ͽ� ����� �� �ֽ��ϴ�.

* [ADADELTA: An Adaptive Learning Rate Method](http://arxiv.org/abs/1212.5701)
* [Stanford CS231n: Optimization Algorithms](http://cs231n.github.io/neural-networks-3/)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)


**ADAGRAD**

Adagrad�� ������ �н��� ��ȭ����Դϴ�. ������ ������ �ð� ��ȭ�� ���� ��� �����ϵ��� �ϴ� ���� Ư¡�̴�. �⺻ SGD�� ����� ����� �� ������, ��� ������ (sparse data)�� ����ϱ� ����. ��� �����Ϳ� ���� �н����� �Ҵ��ϸ� �ұ�Ģ������ �Ķ���Ͱ� ������Ʈ �Ǳ� �����̴�. (�ұ�Ģ���� �����Ϳ� ���ؼ��� ���⼺�� �����Ѵ�)

* [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf)
* [Stanford CS231n: Optimization Algorithms](http://cs231n.github.io/neural-networks-3/)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)


**ADAM**

Adam�� ������ �н��� ��ȭ����Դϴ�. rmsprop�� ����������, ������Ʈ�� ���� ������ 1��, 2�� ���Ʈ, ����(bias)�� �̿��� ��� �����˴ϴ�.

* [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)


**AFFINE LAYER**

���� ��Ʈ��ũ���� ���� ����� ���̾� (fully-connected layer)�Դϴ�. Affine�� ���� ���̾��� �� �������� ���� ���̾��� �� �����鿡 ����Ǿ� ������ ���մϴ�. �������, �̰��� ���� ��Ʈ��ũ�� "ǥ��" ���̾��Դϴ�. Affine layer�� ���� Convolutional Neural Networks�� Recurrent Neural Networks�� ���� ���� ��¿��� ���� ������ �ϱ� ������ �߰��˴ϴ�. Affine layer�� �Ϲ������� y = f(Wx + b)�� ���·� ��Ÿ����, x�� �Է� ���̾�, W�� �Ķ����, b�� ���� (bias), f�� ���� Ȱ��ȭ �Լ��Դϴ�. 


**ATTENTION MECHANISM**

Attention ��Ŀ������ �̹������� Ư�� �κп� �����ϴ� ����� �ð��� �ָ� �ɷ¿� ������ �޾ҽ��ϴ�. Attention ��Ŀ������ �ڿ��� ó���� �̹��� �ν� ��Ű���Ŀ��� ��Ʈ��ũ�� �����ϱ� ���ؼ� �ָ��ؼ� �н��ؾ� �ϴ� �Ϳ� ���� ǥ�� �����մϴ�. 

* [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)


**ALEXNET**

Alexnet�� �̹����� ��ȸ (ILSVRC 2012)�� �̹��� �νĿ��� ���� ū �������̷� ����ߴ�, Convolutional Neural Networks (CNN) �����̸�, CNN�� ������ ��Ȱ��Ų �庻���Դϴ�. 5���� convolutional ���̾�� �� �� �Ϻκ��� max-pooling ���̾�� �����Ǿ� ������, 3���� ���� ����� (fully-connected) ���̾�� 1000���� ����� ���� softmax�� �����մϴ�. Alexnet�� [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)�� �Ұ��Ǿ����ϴ�.


**AUTOENCODER**

Autoencoder�� ���� ��Ʈ��ũ �� �� �Է� ��ü�� �����ϱ� ���� ������ ���� ���Դϴ�. ��Ʈ��ũ ���ο� �������� ("bottleneck")�� �����ϰ� �ִ� ���� Ư¡�Դϴ�. ������������ �Ұ����ڸ�, ��Ʈ��ũ�� �Ͽ��� �Է¿� ���� ������ ǥ������ �н��ϰԲ� �մϴ�. ȿ�������� �Է¿� ���� ���� ǥ�������� �����ϴ� ������. Autoencoder�� PCA�� �ٸ� ���� ��� ����� ������ �ֽ��ϴ�. ������ �������� Ư�������� �� ������ �������� �н��� �� �ֽ��ϴ�. [Denoising Autoencoders](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf), [Variational Autoencoders](http://arxiv.org/abs/1312.6114), [Sequence Autoencoders](http://arxiv.org/abs/1511.01432)�� ���� �پ��� ������ autoencoder ���� �ֽ��ϴ�.

**AVERAGE-POOLING**

Average-Pooling�� �̹��� �ν��� ���� Convolutional Neural Networks�� ���Ǵ� pooling ����� �ϳ��Դϴ�. Ư¡�� ǥ���� ��ġ ���� �����찡 ��ȸ�ϸ� �ش� �������� ��� ���� ����� ���մϴ�. �̴� �Է� ǥ���� �������� ǥ������ �����ϴ� ������ �մϴ�.



B
--------------

**BACKPROPAGATION**

Backpropagation�� ���� ��Ʈ��ũ���� (Ȥ��, �Ϲ����� feedforward �׷�������) ȿ�������� ���(gradient) �� ����ϱ� ���� ����Դϴ�. ��Ʈ��ũ ������� ���� ��̺� ���� ��Ģ�� �̿��� ��絵�� ����Ͽ� �Է������� �����մϴ�. Backpropagation�� ù ����� 1960����� Vapnik�� ��ʷ� �Ž��� �ö󰩴ϴ�. ������ [Learning representations by back-propagating errors](http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html)�� �ٿ��̶�� ���� �ο�ǰ� �ֽ��ϴ�. 

* [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/)


**BACKPROPAGATION THROUGH TIME (BPTT)**

Backpropagation Through Time ([��](http://deeplearning.cs.cmu.edu/pdfs/Werbos.backprop.pdf))�� Backpropagation �˰����� Recurrent Neural Networks (RNNs) �� ����� ���Դϴ�. BPTT�� RNN�� ����Ǵ� ǥ������ backpropagation �˰����̶�� ���� �ֽ��ϴ�. RNN�� �� ������ recurrent ���̾ ��Ÿ����, �� recurrent ���̾�� �Ķ���͸� �����մϴ�. BPTT�� �̸� �״��, RNN�� ��� ������ ���� ���� �Ķ���͸� �����ϱ� ������, �� ������ ������ ��� ���� �������� "�ð��� �Ž��� (through time)" ���޵˴ϴ�. ���鰳�� �Է����� ������ �� sequences�� ó���� ��, ��� ȿ���� ���� truncated BPTT�� ���� ���˴ϴ�. Truncated BPTT�� ������ ���������� ������ �����ϰ� ����ϴ�. 

* [Backpropagation Through Time: What It Does and How to Do It](http://deeplearning.cs.cmu.edu/pdfs/Werbos.backprop.pdf)


**BATCH NORMALIZATION**

Batch Normalization�� ���̾��� �Է��� mini-batch�� ����ȭ �ϱ� ���� ����Դϴ�. �н� �ӵ��� �����ְ�, ���� �н����� ��� �����ϰ� �ϸ�, ����ȭ (regularization) �ϵ��� �մϴ�. Batch Normalization�� Convolutional Neural Networks (CNN)�� Feedforward Neural Networks (FNN)�� ���� ȿ�����̶�� ���������ϴ�. ������ Recurrent Neural Networks ���� ���� ���������� ������� �ʾҽ��ϴ�. 

* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)
* [Batch Normalized Recurrent Neural Networks](http://arxiv.org/abs/1510.01378)


**BIDIRECTIONAL RNN**

Bidirectional Recurrent Neural Network�� �ΰ��� ���� �ٸ� ������ ���� RNN�� �����ϴ� ���� ��Ʈ��ũ�� �ǹ��մϴ�. ������ (forward) RNN�� �Է� sequence�� ó������ ������ �а�, ������ (backward) RNN�� ������ ���� ó���� �������� �н��ϴ�. �� RNN�� �ΰ��� ���� ���ͷ� ���� ������, �Է� ���̾� ��ܿ� �� RNN�� �װ�, �� ���� ��� ���̾�� ���� �˴ϴ�. Bidirectional RNN�� �ڿ��� ó�� �������� ���� ���˴ϴ�. Ư�� �ܾ��� ��, �� �ܾ��� �ǹ̸� ���� ���� �ܾ �����ϴ� ������ ����˴ϴ�.

* [Bidirectional Recurrent Neural Networks](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)



C
--------------

**CAFFE**

[Caffe](http://caffe.berkeleyvision.org/)�� Berkeley Vision and Learning Center���� ���ߵ� ������ �����ӿ�ũ�Դϴ�. Caffe�� Ư�� �������� ������ CNN ���� ó���ϴµ� Ưȭ�Ǿ� �ֽ��ϴ�. 


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