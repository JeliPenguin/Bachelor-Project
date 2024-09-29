# Learning To Communicate Under Noise: Multiagent Reinforcement Learning for Cooperation and Coordination under Noisy Communication

## Abstract

A novel MARL algorithm is introduced to tackle the Guide and Scout Grid world problem, emphasizing effective cooperation and coordination among Scout agents while having to communicate under a noisy communication channel. To assess the algorithm's effectiveness, rigorous testing and analysis are conducted in two fully cooperative instances of the problem: Finding-Treat and Spread both using Binary Symmetric Channel with parameter 𝓅.

## Libraries and Hardware

To install the libraries required for this project, use the command pip install -r requirements.txt

Note that you must have CUDA installed and a suitable GPU to be able to run the pre-trained models.

## How to use

The major functions are located in main.py for tuning hyperparameters and for evaluating results. To start a brand new training, you could refer to the example quickTrain function that shows how training settings and environment can be personalised.

To evaluate the results, instantiate an Evaluator object can call the evaluate method, reeval can be set to true to re-evaluate models that has already been evaluated before. Results of all evaluation can be viewed by calling the plotAll() method. To see the top performing models, call plotBest(). For comparison of best model with Checksum, Norm and Norm_Noised, call normNoiseCompare() method
