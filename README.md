# Deep Learning Replications

These are a mixture of partial replications of landmark papers in deep learning research and small investigative projects inspired by those papers. 

In many cases, the projects follow recommendations Jacob Hilton makes in his Deep Learning Curriculum, which is available athttps://github.com/jacobhilton/deep_learning_curriculum

All of them currently work with the MPS device in the latest pytorch release.

## Projects
\#1 Scaling Laws 

This is a test following Kaplan et. al.'s "Scaling Laws for Neural Language Models" and Hoffman et. al.'s "Training Compute-Optimal Large Language Models". Since I don't have access to enough compute to estimate the relation between loss, model size, and data size for language models, I've followed Hilton's recommendation of using a small conv net with the MNIST data set. To allow for more variation among the measured data sizes, I copy and randomly transform the MNIST data set with rotations and distortions several times over. 

To Do: estimate formulae for optimal model size and data size given fixed compute availability. 

\#2 Transformers

This is a decoder-only implementation of the transformer model introduced in Vaswani et. al.'s "Attention is All You Need." I train the model on sequences of ascending integers, sequences of random integers that repeat in the second half of the sequence (e.g. [1, 5, 2, 4, 0, 1, 5, 2, 4]), Shakespeare, and the King James Bible. Performance on the Shakespeare and KJV modeling tasks is currently pretty low. One part of the project is to tweak the model and hyperparameters to improve the language modeling performance.

The second part of this project is investigating the extent to which McCandlish et. al.'s theory and empirical findings in "An Empirical Model of Large-Batch Training" apply to small instances of the transformer architecture. 

\#3 Reinforcement Learning

This is a basic implementation of VPG. I validated the implementation on a few OpenAI Gym environments and am currently working on getting the network to perform in more difficult Procgen environments. 
