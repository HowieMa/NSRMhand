# NSRMhand
Pytorch implementation of WACV 2020 paper Nonparametric Structure Regularization Machine for 2D Hand Pose Estimation



## Abstract
Hand pose estimation is more challenging than body pose estimation due to severe articulation, self-occlusion and high dexterity of the hand. Current approaches often rely on a popular body pose algorithm, such as the Convolutional Pose Machine (CPM), to learn 2D keypoint features. These algorithms cannot adequately address the unique challenges of hand pose estimation, because they are trained solely based on keypoint positions without seeking to explicitly model structural relationship between them. We propose a novel Nonparametric Structure Regularization Machine (NSRM) for 2D hand pose estimation, adopting a cascade multi-task architecture to learn hand structure and keypoint representations jointly. The structure learning is guided by synthetic hand mask representations, which are directly computed from keypoint positions, and is further strengthened by a novel probabilistic representation of hand limbs and an anatomically inspired composition strategy of mask synthesis. We conduct extensive studies on two public datasets - OneHand 10k and CMU Panoptic Hand. Experimental results demonstrate that explicitly enforcing structure learning consistently improves pose estimation accuracy of CPM baseline models, by 1.17% on the ﬁrst dataset and 4.01% on the second one.

Visualization of our proposed LDM-G1, LPM-G1, and our network structure.    
![LPM G1](readme/ldm_g1.jpg) 
![LDM G6](readme/lpm_g1.jpg)


![net](readme/net.jpeg)



## Highlights
- We propose a novel cascade structure regularization methodology for 2D hand pose estimation, 
which utilizes synthetic hand masks to guide keypoints structure learning.
- We propose a novel probabilistic representation of hand limbs and an anatomically inspired composition strategy for hand mask synthesis.

## Running
0. Prepare  
 ~~~ 
pytorch >= 1.0  
torchvision >= 0.2 
numpy  
matplotlib 
~~~

1. Download our formated Panoptic dataset or format your own dataset based on `data_sample/`. 

2. Specify your configuration in configs/xxx.json.  
You can also use the default parameter settings, but remember to change the **data root**.  


3. Train model by 
~~~
python main.py + xxx.json
~~~

For example, if you want to train LPM G1, you should run 
~~~
python main.py LPM_G1.json
~~~


## Notation
- The most creativie part of our model is the structure representation, which is generated from keypoints only.  
you can see `dataset/hand_ldm.py` and `dataset/hand_lpm.py` for detail and adapt it for other tasks.

- Since this is a multi task learning problem, 
the weight ande decay ratio of keypoint confidence map loss and NSRM loss may vary for different dataset, 
you may need to adjust these parameters for your own dataset. 

- In our experiments and code, we only apply our NSRM to CPM, but we believe it will also work for other pose estimation network, 
such as Stacked Hourglass, HR-Net.  

