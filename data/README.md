# Cora Citation Network Dataset

This folder contains the **Cora citation network dataset**, widely used for **Graph Neural Network (GNN)** research and **node classification** tasks.

---

## Overview

The **Cora dataset** consists of **machine learning research papers**, each classified into one of **seven categories**:

- Case_Based  
- Genetic_Algorithms  
- Neural_Networks  
- Probabilistic_Methods  
- Reinforcement_Learning  
- Rule_Learning  
- Theory

Each paper (node) either **cites** or **is cited by** at least one other paper, forming a connected citation network.

---

## Dataset Statistics

| Property | Description |
|-----------|--------------|
| **Number of Nodes** | 2,708 papers |
| **Number of Edges** | 5,429 citations |
| **Number of Features per Node** | 1,433 |
| **Number of Classes** | 7 |
| **Graph Type** | Citation Network |

---

## Files in This Directory

This directory includes **two main files**:

### 1. `cora.content`

Each line represents a paper and its features in the following format:

```
<paper_id> <word_attributes>+ <class_label>
```

- **paper_id** → unique identifier for each paper  
- **word_attributes** → binary indicators (1 if the word is present, 0 otherwise) for 1,433 vocabulary words  
- **class_label** → the research topic of the paper  

**Example:**

```
31336 0 1 0 0 ... 0 Neural_Networks
```

---

### 2. `cora.cites`

Each line describes a citation relationship between two papers:

```
<ID of cited paper> <ID of citing paper>
```

- The **first ID** is the paper being *cited*  
- The **second ID** is the paper *doing the citing*  
- Direction of citation: **right → left**, meaning `"paper1 paper2"` corresponds to **paper2 → paper1**

**Example:**

```
paper1 paper2
```
means **paper2 cites paper1**.

---

## Usage Example

Example code to load the dataset:

```python
from utils import load_data

adj, features, labels, idx_train, idx_val, idx_test = load_data(path="./data/cora/", dataset="cora")
```

**Returned objects:**
- `adj` → adjacency matrix (graph structure)  
- `features` → node feature matrix  
- `labels` → class labels  
- `idx_train`, `idx_val`, `idx_test` → index splits for training, validation, and testing  

---

## Reference

Original source: [Cora Dataset](https://graphsandnetworks.com/the-cora-dataset/)

---
📁 **Files Required**
- `cora.content`  
- `cora.cites`
