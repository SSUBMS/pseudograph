🏗️ 프로젝트 구조
GAT-HiC/
├── README.md
├── requirements.txt
├── setup.sh                   # 환경 설정 스크립트
├── run_pipeline.sh            # 전체 파이프라인 실행
├── check_structure.py         # 프로젝트 구조 검증
│
├── Core Pipeline Scripts
│   ├── hic_to_matrix.py          # Hi-C 데이터 → Adjacency Matrix 변환
│   ├── node2vec_embeddings.py    # Node2Vec 임베딩 생성
│   ├── ground_truth.py           # Ground Truth 생성
│   ├── graph_constructor.py      # PyTorch Geometric 그래프 데이터 생성
│   └── model_trainer.py          # GAT 모델 학습 및 평가
│
├── Analysis Tools
│   └── analyze_results.py        # 결과 분석 및 시각화
│
├── data/
│   ├── Raw Hi-C Data
│   │   ├── bulk_1.txt                      # Bulk Hi-C 데이터 (raw)
│   │   ├── GM12878_bulk_1_23.42%.txt       # GM12878 cell line
│   │   ├── H1Esc_bulk_1_38.03%.txt         # H1Esc cell line
│   │   ├── HAP1_bulk_1_19.05%.txt          # HAP1 cell line
│   │   ├── HFF_bulk_1_17.36%.txt           # HFF cell line
│   │   └── IMR90_bulk_1_2.14%.txt          # IMR90 cell line
│   │
│   ├── Processed Data
│   │   ├── *_matrix.txt                    # Adjacency matrices
│   │   ├── *_embeddings.txt                # Node2Vec embeddings
│   │   ├── ground_truth_data.pkl           # Ground truth labels
│   │   └── graph_data.pkl                  # PyTorch Geometric data
│
└── results/
    ├── best_model.pth                      # 최고 성능 모델
    ├── training_history.png                # 학습 곡선
    ├── training_results.pkl                # 학습 결과
    └── Analysis Plots
        ├── prediction_quality_analysis.png
        └── cell_line_analysis.png
🔄 파이프라인 워크플로우
Phase 1: 데이터 준비
1. Hi-C 데이터 → Adjacency Matrix 변환
bashpython hic_to_matrix.py data/

입력: *_bulk.txt (List format: row, col, value)
출력: *_matrix.txt (NxN adjacency matrix)
기능:

List format Hi-C 데이터를 adjacency matrix로 변환
대칭 행렬 생성 (Hi-C는 symmetric)
Self-loops 제거 (대각선 0으로 설정)
통계 정보 저장 (*_stats.txt)



2. Node2Vec 임베딩 생성
bashpython node2vec_embeddings.py data/

입력: *_matrix.txt
출력: *_embeddings.txt (Nx512 embedding matrix)
파라미터:

dimensions: 512
walk_length: 150
num_walks: 50
p: 1.75 (return parameter)
q: 0.4 (in-out parameter)
window: 25



3. Ground Truth 생성
bashpython ground_truth.py

입력:

bulk_1_matrix.txt (Bulk Hi-C matrix)
*_bulk_1_*%_matrix.txt (5개 cell line matrices)


출력:

ground_truth_data.pkl (학습용 레이블)
ground_truth_data.csv (CSV 형태)


기능:

각 엣지에 대한 cell line별 기여도 비율 계산
Formula: ratio_cellline_A = Raw_A[i,j] / Raw_Bulk[i,j]
데이터 일관성 검증 (비율의 합이 1인지 확인)



4. PyTorch Geometric 그래프 데이터 생성
bashpython graph_constructor.py

입력:

bulk_1_kr_matrix.txt (KR 정규화된 contact matrix)
bulk_1_kr_embeddings.txt (Node2Vec 임베딩)


출력: graph_data.pkl (PyTorch Geometric Data 객체)
구성 요소:

x: 노드 특징 (Node2Vec 임베딩, Nx512)
edge_index: 엣지 연결 정보 (2 x num_edges)
edge_attr: 엣지 가중치 (contact frequency)
num_nodes: 노드 수



Phase 2: 모델 학습 및 평가
5. 모델 학습
bashpython model_trainer.py
모델 아키텍처:
Input Graph Data
    ↓
[GAT Encoder] (3 layers, 4 attention heads)
    - Layer 1: 512 → 512 (hidden_dim * num_heads)
    - Layer 2: 512 → 512
    - Layer 3: 512 → 64 (output_dim)
    ↓
Node Embeddings (N x 64)
    ↓
[Edge Decoder] (MLP, 3 layers)
    - Concat: z_i || z_j → 128 dims
    - Hidden: 128 → 128
    - Output: 128 → 5 (cell lines)
    - Softmax activation
    ↓
Predicted Ratios (5 probabilities per edge)
학습 과정:

Ground truth에서 엣지와 해당 레이블 로드
Train/Val/Test split (70/15/15)
미니배치 학습 (batch_size=512)
KL-Divergence Loss 최소화:

python   Loss = KL(P_true || P_pred) = Σ P_true(i) * log(P_true(i) / P_pred(i))

Early stopping (patience=10)
