# GAT-HiC: Cell Line Contribution Estimation

Graph Attention Network를 이용한 Bulk Hi-C 데이터의 Cell Line별 기여도 예측

---

## 📋 개요

혼합된 Bulk Hi-C 데이터에서 각 contact이 어떤 cell line (GM12878, H1Esc, HAP1, HFF, IMR90)에서 유래했는지 **기여도 비율을 예측**하는 모델

**핵심 아이디어**: 
- Graph Neural Network (GAT)로 Hi-C contact 네트워크를 학습
- 각 edge에 대해 5개 cell line의 기여도 비율 예측 (합=1)
- Supervised learning 방식 (Ground truth 필요)

---

## 🏗️ 프로젝트 구조

```
GAT-HiC/
├── hic_to_matrix.py          # Hi-C 데이터 → Adjacency Matrix
├── node2vec_embeddings.py    # Node2Vec 임베딩 생성
├── ground_truth.py           # Ground Truth 레이블 생성
├── graph_constructor.py      # PyTorch Geometric 그래프 데이터
└── model_trainer.py          # GAT 모델 학습 및 평가
```

---

## 🔄 실행 순서

```bash
# 1. Hi-C 데이터 → Adjacency Matrix 변환
python hic_to_matrix.py data/

# 2. Node2Vec 임베딩 생성 (각 노드의 구조적 특징)
python node2vec_embeddings.py data/

# 3. Ground Truth 생성 (각 edge의 cell line별 비율)
python ground_truth.py

# 4. PyTorch Geometric 그래프 데이터 생성
python graph_constructor.py

# 5. GAT 모델 학습
python model_trainer.py
```

---

## 📊 파이프라인

```
Raw Hi-C Data
     ↓
[hic_to_matrix.py] → Adjacency Matrix (NxN)
     ↓
[node2vec_embeddings.py] → Node Embeddings (Nx512)
     ↓
[ground_truth.py] → Edge Labels (cell line ratios)
     ↓
[graph_constructor.py] → PyTorch Geometric Data
     ↓
[model_trainer.py] → Trained GAT Model
```

---

## 🔧 각 스크립트 설명

### 1. hic_to_matrix.py
**입력**: `*_bulk.txt` (row, col, value 형식)  
**출력**: `*_matrix.txt` (NxN adjacency matrix)  
**기능**: List format Hi-C를 대칭 행렬로 변환

### 2. node2vec_embeddings.py
**입력**: `*_matrix.txt`  
**출력**: `*_embeddings.txt` (Nx512)  
**기능**: Random walk 기반 노드 임베딩 생성 (구조적 특징 포착)

**파라미터**: dimensions=512, walk_length=150, num_walks=50, p=1.75, q=0.4

### 3. ground_truth.py
**입력**: 
- `bulk_1_matrix.txt`
- `GM12878_bulk_1_23.42%_matrix.txt` (5개 cell line)

**출력**: 
- `ground_truth_data.pkl` (학습용)
- `ground_truth_data.csv` (분석용)

**기능**: 각 edge (i,j)에 대해 cell line별 기여도 비율 계산
```python
ratio_cellline_A = Raw_A[i,j] / Raw_Bulk[i,j]
```

### 4. graph_constructor.py
**입력**: 
- `bulk_1_kr_matrix.txt` (KR normalized)
- `bulk_1_kr_embeddings.txt`

**출력**: `graph_data.pkl` (PyTorch Geometric Data)

**구성**:
- `x`: Node features (Nx512)
- `edge_index`: Edge connections (2 x E)
- `edge_attr`: Edge weights
- `num_nodes`: N

### 5. model_trainer.py
**입력**: `graph_data.pkl`, `ground_truth_data.pkl`  
**출력**: `best_model.pth`, `training_results.pkl`

**모델 구조**:
```
Graph (N nodes, E edges)
    ↓
GAT Encoder (3 layers, 4 heads)
    512 → 512 → 64
    ↓
Node Embeddings (Nx64)
    ↓
Edge Decoder (MLP)
    Concat(z_i, z_j) → 128 → 128 → 5
    Softmax
    ↓
5 cell line ratios per edge
```

**학습**: KL-Divergence Loss, Adam optimizer, Early stopping

---

## 📦 필수 패키지

```bash
pip install torch torch-geometric node2vec networkx numpy pandas scikit-learn matplotlib
```

**주요 버전**:
- `torch>=2.0.0`
- `torch-geometric>=2.3.0`
- `node2vec>=0.4.6`

---

## 📂 데이터 형식

### 입력 데이터
```
data/
├── bulk_1.txt                      # Bulk Hi-C (raw)
├── GM12878_bulk_1_23.42%.txt       # Cell line 1
├── H1Esc_bulk_1_38.03%.txt         # Cell line 2
├── HAP1_bulk_1_19.05%.txt          # Cell line 3
├── HFF_bulk_1_17.36%.txt           # Cell line 4
└── IMR90_bulk_1_2.14%.txt          # Cell line 5
```

**Hi-C raw format** (`.txt`):
```
# row col value
0 0 98388
0 1 21212
0 2 5413
...
```

### 중간 데이터
- `*_matrix.txt`: NxN adjacency matrix
- `*_embeddings.txt`: Nx512 node embeddings
- `ground_truth_data.pkl`: Edge labels (ratios)
- `graph_data.pkl`: PyTorch Geometric Data

### 출력 결과
- `best_model.pth`: 학습된 모델
- `training_results.pkl`: 성능 지표


---

**개발자**: 김찬주 (Chanju Kim)  
**소속**: Biomedical Data Science Laboratory  
**버전**: 1.0.0 (2025-02-01)
