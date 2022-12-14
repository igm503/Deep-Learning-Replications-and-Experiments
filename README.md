# Deep Learning Replications

These are a mixture of partial replications of landmark papers in deep learning research and small investigative projects inspired by those papers. 

In many cases, the projects follow recommendations Jacob Hilton makes in his Deep Learning Curriculum, which is available at https://github.com/jacobhilton/deep_learning_curriculum

All of them currently work with the MPS device in the latest pytorch release.

## Projects
### 1 Scaling Laws 

This is a test following Kaplan et. al.'s "Scaling Laws for Neural Language Models" and Hoffman et. al.'s "Training Compute-Optimal Large Language Models". Since I don't have access to enough compute to estimate the relation between loss, model size, and data size for language models, I've followed Hilton's recommendation of using a small conv net with the MNIST data set. 

I also test with an MLP model and perform another test for both model types with the EMNIST dataset with elastic distortions to try to approximate a situation in which more than the 60,000 MNIST images are available. In all cases, the models scale much more with size than with data, but the limited amount of unique data makes me suspicious of these results. 

### 2 Transformers

This is a decoder-only implementation of the transformer model introduced in Vaswani et. al.'s "Attention is All You Need." I train the model on sequences of ascending integers, sequences of random integers that repeat in the second half of the sequence (e.g. [1, 5, 2, 4, 0, 1, 5, 2, 4]), Shakespeare, and the King James Bible. Performance on the Shakespeare and KJV modeling tasks is currently pretty low. One part of the project is to tweak the model and hyperparameters to improve the language modeling performance.

The second part of this project is investigating the extent to which McCandlish et. al.'s theory and empirical findings in "An Empirical Model of Large-Batch Training" apply to small instances of the transformer architecture. 

### 3 Reinforcement Learning

This is a basic implementation of VPG and PPO (the latter as described in Schulman et. al.'s "Proximal Policy Optimization Algorithms"). I validated the implementations on a few OpenAI Gym environments and am currently working on getting the PPO-trained network to perform in more difficult Procgen environments. 
