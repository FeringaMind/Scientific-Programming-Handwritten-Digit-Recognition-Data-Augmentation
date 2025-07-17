# Handwritten Digit Recognition – Data Augmentation

***This Project was created and maintained as part of the Scientific Programming course (B.Sc CS) @TUHH***



## Information

The objective of this project is to develop and evaluate a handwritten digit classifier using the MNIST dataset. Rather than trying diﬀerent architectures, we want you to experiment with how the training data aﬀects the classifier’s performance. You don’t need deep knowledge of machine learning to succeed in the project, but it does require some eﬀort on your part to get used to some basic concepts of machine learning. More specifically, the project includes the following tasks:

1. Download the MNIST dataset either from http://yann.lecun.com/exdb/mnist/ or using the Julia package MLDatasets.jl.

2. Write a method to train a simple convolutional neural network called LeNet to classify the images. To get started, check out the Julia machine learning library Flux.jl. We also encourage you to look at the short guides provided by Flux.jl at https://fluxml.ai/Flux.jl/stable/guide/models/quickstart/ and the following pages. For the following tasks, keep the network, the training algorithm, and other hyperparameters fixed and concentrate on the data instead.

3. Write methods to evaluate the performance of your classifier. What accuracy do you achieve on the test dataset? Do you get the same accuracy in practice? If not, what might be the reasons?

4. Train the classifier using diﬀerent proportions of the original training data. How does the size of the training set aﬀect the classifier’s performance?

5. Using a fraction of the original training set (e.g., 10%), implement and apply diﬀerent data augmentation methods to improve the classifier’s performance. Can you achieve a performance similar to or better than that of a network trained on the full training set?



## Project Schedule

- Starting point is the kickoﬀ meeting (2025-06-20)

- Project phase with weekly meetings (2025-06-27, 2025-07-04, and 2025-07-11) with your supervisor

- Presentations will finalize the project (2025-07-18)



## Roadmap

| Open ⬜ | Done ✅ | In progress ⚙️ | Not working ❌ | Other ❓ |
| ------ | ------ | ------------- | ------------- | ------- |

- [x] **2025-06-20 to 2025-06-27**: Preparation  
    ✅ Readme.md  
    ✅ Roadmap  
    ✅ Setup Project

- [x] **2025-06-27**: Kickoff meeting w/ group and supervisor

- [x] **2025-06-27 to 2025-07-04**: Sprint 1  
    ✅ Repository Cleanup and writing train! function ⭐⭐⭐ *(refers to 2)*  
    ✅ Write confusion matrix and accuracy function ⭐⭐⭐ *(refers to 3)*

- [x] **2025-07-05 to 2025-07-11**: Sprint 2  
    ✅ Save training models ⭐⭐ *(refers to 4)*  
    ✅ Evaluation/Discussion/Testing ⭐⭐⭐ *(refers to 4/5)*  
    ✅ Fix Augmentation and Backend ⭐⭐ *(refers to 5)*

- [x] **2025-07-12 to 2025-07-16**: Sprint 3  
    ✅ Writing unit tests ⭐⭐ *(refers to general tasks)*  
	✅ Rework Frontend text⭐⭐ *(refers to general tasks)*
    ✅ Preparing the presentation ⭐⭐ *(refers to general tasks)*


- [ ] **2025-07-18**: Presentation



## Directory

## 📁 Projektstruktur

```
Scientific-Programming-Handwritten-Digit-Recognition-Data-Augmentation
├── 📁 models
│   └── model_54210.bson
├── 📁 src
│   ├── augmentation_backend.jl
│   └── model_backend.jl
├── 📁 tests
│   └── test_LeNet5.jl
├── .gitignore
├── FrontendPluto.jl
├── Manifest.toml
├── Project.toml
└── README.md
```


## Contributers

Paul Hain - paulhain.developer@gmail.com

Ilya Acik - ilyaacik.dev@gmail.com

Yusa Kaya - yusakaya.dev@gmail.com

