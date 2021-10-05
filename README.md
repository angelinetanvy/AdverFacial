# AdverFacial

## Abstract

The objective of AdverFacial is to protect face's privacy on video conferencing, especially in this era of pandemic, where video calling is the platform used to replace all face-to-face activities. This is achieved by conducting experiment on fooling state-of-the-art classifiers on micro-expression datasets. If by applying perturbation on the datasets successfully decrease the accuracy of the classifier, the experiment is a succeed.

## Resources and References
The datasets used are as follows:
- CASME (doi:10.1109/FG.2013.6553799)
- CASME2 (doi:10.1371/journal.pone.0086041)
- CAS(ME)^2 (doi:10.1109/TAFFC.2017.2654440)
- SMIC (doi:10.1109/FG.2013.6553717)

The pretrained classifiers are cited from 2 repositories, which are stated below:
- LEARNet (https://visionintelligence.github.io/request_FER.html)
- STCNN (https://github.com/bogireddytejareddy/micro-expression-recognition)

Universal Adversarial Perturbation:
- Deep Fool and Project Perturbation (https://github.com/NetoPedro/Universal-Adversarial-Perturbations-Pytorch)

# Content
Divided by datasets, each dataset folder contains the application of 2 models, STCNN and LEARNet being fed with the corresponding dataset. This is to gain the baseline accuracy of each dataset with each model. Universal Adversarial Perturbation is also applied to each of the models for each datasets, in order to gain the fooling rate of it.

## Prerequisites
- [TensorFlow 2.5.0]
- [Keras 2.0.0]


## Citation

@article{reddy2019microexpression,
	title={Spontaneous Facial Micro-Expression Recognition using 3D Spatiotemporal Convolutional Neural Networks},
	author={Sai Prasanna Teja, Reddy and Surya Teja, Karri and Dubey, Shiv Ram and Mukherjee, Snehasis},
	journal={International Joint Conference on Neural Networks},
	year={2019}
	}
    
@InProceedings{Moosavi_Dezfooli_2017_CVPR,
	author = {Moosavi-Dezfooli, Seyed-Mohsen and Fawzi, Alhussein and Fawzi, Omar and Frossard, Pascal},
	title = {Universal Adversarial Perturbations},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {July},
	year = {2017}
	}
	
@Article{verma_vipparthi_singh_murala_2020, 
	title={LEARNet: Dynamic Imaging Network for Micro Expression Recognition}, 
	volume={29}, DOI={10.1109/tip.2019.2912358}, 
	journal={IEEE Transactions on Image Processing}, 
	author={Verma, Monu and Vipparthi, Santosh Kumar and Singh, Girdhari and Murala, Subrahmanyam}, 
	year={2020}, 
	pages={1618–1627}
	}
    
   
