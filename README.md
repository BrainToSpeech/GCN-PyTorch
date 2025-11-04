# Graph Convolutional Networks (GCN) in PyTorch
- This is an implementation of **Graph Convolutional Networks (GCNs)** for node classification on the **Cora citation dataset**.
---

## Setup
Before running the code, make sure you have installed all required dependencies.
You can easily install them using the provided 'requirements.txt' file:
```commandline
pip install -r requirements.txt
```
> This code is tested with Python 3.11.9
---

## Dataset: Cora

- **Nodes:** Scientific publications  
  Each node represents an academic paper.

- **Edges:** Citation relationships  
  An edge exists between two papers if one cites the other.

- **Node Features:** Word vectors  
  Each publication is represented by a 1,433-dimensional bag-of-words feature vector.

- **Labels (7 Classes):** Publication subjects  
  The Cora dataset contains 2,708 papers classified into 7 research categories.

---

## Dataset Summary

| Property | Description |
|-----------|--------------|
| **Number of Nodes** | 2,708 |
| **Number of Edges** | 5,429 |
| **Number of Features per Node** | 1,433 |
| **Number of of Classes** | 7 |
| **Graph Type** | Citation Network |

---

## Overview

This project demonstrates how **Graph Convolutional Networks (GCNs)** can learn node representations by leveraging both **graph structure** and **node features**.  
The objective is to classify each publication into its corresponding research topic based on citation relationships.

---

## Reference

> Thomas N. Kipf and Max Welling.  
> *Semi-Supervised Classification with Graph Convolutional Networks.*  
> ICLR 2017. [[Paper Link]](https://arxiv.org/abs/1609.02907)

---
