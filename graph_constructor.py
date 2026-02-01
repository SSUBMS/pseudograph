import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from pathlib import Path
import logging
import pickle
from typing import Tuple, Optional
import scipy.sparse as sp

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphConstructor:
    def __init__(self, data_path: str):
        """
        GNN 입력 그래프 구성기 초기화
        
        Args:
            data_path (str): 데이터 파일들이 있는 경로
        """
        self.data_path = Path(data_path)
        
        # 파일 경로 설정
        self.kr_matrix_file = self.data_path / "bulk_1_kr_matrix.txt"
        self.kr_embeddings_file = self.data_path / "bulk_1_kr_embeddings.txt"
        
        # 데이터 저장용
        self.kr_matrix = None
        self.node_embeddings = None
        self.edge_index = None
        self.edge_attr = None
        self.graph_data = None
    
    def load_kr_matrix(self) -> np.ndarray:
        """
        KR 정규화된 Contact Matrix를 로드합니다.
        
        Returns:
            np.ndarray: KR 정규화된 Contact Matrix
        """
        try:
            matrix = np.loadtxt(self.kr_matrix_file)
            logger.info(f"Loaded KR matrix: shape {matrix.shape}")
            
            # NaN 값 확인 및 처리
            nan_count = np.isnan(matrix).sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in KR matrix. Replacing with 0.")
                matrix = np.nan_to_num(matrix, nan=0.0)
            
            # 무한대 값 확인 및 처리
            inf_count = np.isinf(matrix).sum()
            if inf_count > 0:
                logger.warning(f"Found {inf_count} infinite values in KR matrix. Replacing with 0.")
                matrix = np.nan_to_num(matrix, posinf=0.0, neginf=0.0)
            
            return matrix
            
        except Exception as e:
            logger.error(f"Error loading KR matrix from {self.kr_matrix_file}: {e}")
            raise
    
    def load_node_embeddings(self) -> np.ndarray:
        """
        Node2Vec 임베딩을 로드합니다.
        
        Returns:
            np.ndarray: 노드 임베딩 행렬 (n_nodes x embedding_dim)
        """
        try:
            embeddings = np.loadtxt(self.kr_embeddings_file)
            logger.info(f"Loaded node embeddings: shape {embeddings.shape}")
            
            # NaN 값 확인 및 처리
            nan_count = np.isnan(embeddings).sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in embeddings. Replacing with 0.")
                embeddings = np.nan_to_num(embeddings, nan=0.0)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings from {self.kr_embeddings_file}: {e}")
            raise
    
    def create_edge_index(self, threshold: float = 0.0, top_k_per_node: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Contact Matrix로부터 그래프의 edge_index와 edge_attr을 생성합니다.
        
        Args:
            threshold (float): 엣지 생성을 위한 최소 contact 값 임계값
            top_k_per_node (Optional[int]): 각 노드당 최대 연결 수 (None이면 모든 임계값 이상 연결)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (edge_index, edge_attr)
        """
        logger.info(f"Creating edge index with threshold={threshold}, top_k={top_k_per_node}")
        
        n_nodes = self.kr_matrix.shape[0]
        
        if top_k_per_node is not None:
            # 각 노드당 상위 k개 연결만 유지
            edge_list = []
            edge_weights = []
            
            for i in range(n_nodes):
                # i번째 노드의 모든 연결 강도
                connections = self.kr_matrix[i, :]
                
                # 자기 자신과의 연결 제외
                connections[i] = -np.inf
                
                # 상위 k개 인덱스 찾기
                if top_k_per_node < n_nodes:
                    top_k_indices = np.argpartition(connections, -top_k_per_node)[-top_k_per_node:]
                    # 실제로 임계값보다 큰 것들만 필터링
                    valid_indices = top_k_indices[connections[top_k_indices] > threshold]
                else:
                    valid_indices = np.where(connections > threshold)[0]
                
                for j in valid_indices:
                    edge_list.append([i, j])
                    edge_weights.append(self.kr_matrix[i, j])
        
        else:
            # 임계값 이상의 모든 연결 유지
            edge_list = []
            edge_weights = []
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and self.kr_matrix[i, j] > threshold:
                        edge_list.append([i, j])
                        edge_weights.append(self.kr_matrix[i, j])
        
        if len(edge_list) == 0:
            logger.warning("No edges found with the given threshold. Creating minimal graph.")
            # 최소한의 연결성을 위해 각 노드를 다음 노드와 연결
            edge_list = [[i, (i+1) % n_nodes] for i in range(n_nodes)]
            edge_weights = [1.0] * len(edge_list)
        
        # 양방향 그래프로 만들기 (undirected graph)
        edge_list_undirected = []
        edge_weights_undirected = []
        
        for idx, (i, j) in enumerate(edge_list):
            edge_list_undirected.extend([[i, j], [j, i]])
            weight = edge_weights[idx]
            edge_weights_undirected.extend([weight, weight])
        
        edge_index = torch.tensor(edge_list_undirected, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights_undirected, dtype=torch.float)
        
        logger.info(f"Created edge_index: {edge_index.shape} (num_edges: {edge_index.shape[1]})")
        logger.info(f"Graph density: {edge_index.shape[1] / (n_nodes * (n_nodes - 1)):.6f}")
        
        return edge_index, edge_attr
    
    def normalize_edge_attr(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        엣지 속성을 정규화합니다.
        
        Args:
            edge_attr (torch.Tensor): 원본 엣지 속성
            
        Returns:
            torch.Tensor: 정규화된 엣지 속성
        """
        # Min-Max 정규화
        min_val = edge_attr.min()
        max_val = edge_attr.max()
        
        if max_val > min_val:
            normalized = (edge_attr - min_val) / (max_val - min_val)
        else:
            normalized = torch.ones_like(edge_attr)
        
        logger.info(f"Edge attributes normalized: min={normalized.min():.4f}, max={normalized.max():.4f}")
        return normalized
    
    def create_graph_data(self, 
                         threshold: float = 0.01, 
                         top_k_per_node: Optional[int] = 100,
                         normalize_edges: bool = True) -> Data:
        """
        PyTorch Geometric Data 객체를 생성합니다.
        
        Args:
            threshold (float): 엣지 생성 임계값
            top_k_per_node (Optional[int]): 각 노드당 최대 연결 수
            normalize_edges (bool): 엣지 속성 정규화 여부
            
        Returns:
            Data: PyTorch Geometric Data 객체
        """
        logger.info("Creating PyTorch Geometric Data object...")
        
        # 1. 데이터 로드
        self.kr_matrix = self.load_kr_matrix()
        self.node_embeddings = self.load_node_embeddings()
        
        # 2. 노드 특징과 행렬 크기 일치 확인
        n_nodes_matrix = self.kr_matrix.shape[0]
        n_nodes_embeddings = self.node_embeddings.shape[0]
        
        if n_nodes_matrix != n_nodes_embeddings:
            logger.error(f"Size mismatch: matrix has {n_nodes_matrix} nodes, "
                        f"embeddings have {n_nodes_embeddings} nodes")
            raise ValueError("Matrix and embeddings size mismatch")
        
        # 3. 엣지 인덱스 생성
        self.edge_index, self.edge_attr = self.create_edge_index(
            threshold=threshold, 
            top_k_per_node=top_k_per_node
        )
        
        # 4. 엣지 속성 정규화
        if normalize_edges:
            self.edge_attr = self.normalize_edge_attr(self.edge_attr)
        
        # 5. 노드 특징을 PyTorch 텐서로 변환
        node_features = torch.tensor(self.node_embeddings, dtype=torch.float)
        
        # 6. Data 객체 생성
        self.graph_data = Data(
            x=node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=n_nodes_matrix
        )
        
        logger.info(f"Graph Data created successfully:")
        logger.info(f"  - Nodes: {self.graph_data.num_nodes}")
        logger.info(f"  - Edges: {self.graph_data.num_edges}")
        logger.info(f"  - Node features: {self.graph_data.x.shape}")
        logger.info(f"  - Edge attributes: {self.graph_data.edge_attr.shape}")
        
        return self.graph_data
    
    def save_graph_data(self, output_file: str = "graph_data.pkl"):
        """
        생성된 그래프 데이터를 저장합니다.
        
        Args:
            output_file (str): 저장할 파일명
        """
        if self.graph_data is None:
            raise ValueError("No graph data to save. Run create_graph_data first.")
        
        output_path = self.data_path / output_file
        
        save_data = {
            'graph_data': self.graph_data,
            'metadata': {
                'num_nodes': self.graph_data.num_nodes,
                'num_edges': self.graph_data.num_edges,
                'node_feature_dim': self.graph_data.x.shape[1],
                'kr_matrix_file': str(self.kr_matrix_file),
                'embeddings_file': str(self.kr_embeddings_file)
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Graph data saved to {output_path}")
    
    def get_graph_statistics(self):
        """
        그래프의 통계 정보를 출력합니다.
        """
        if self.graph_data is None:
            logger.warning("No graph data available. Run create_graph_data first.")
            return
        
        data = self.graph_data
        
        print("\n=== Graph Statistics ===")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Node feature dimension: {data.x.shape[1]}")
        print(f"Average degree: {data.num_edges / data.num_nodes:.2f}")
        
        # 노드 특징 통계
        print(f"\nNode features statistics:")
        print(f"  Min: {data.x.min():.4f}")
        print(f"  Max: {data.x.max():.4f}")
        print(f"  Mean: {data.x.mean():.4f}")
        print(f"  Std: {data.x.std():.4f}")
        
        # 엣지 속성 통계
        if data.edge_attr is not None:
            print(f"\nEdge attributes statistics:")
            print(f"  Min: {data.edge_attr.min():.4f}")
            print(f"  Max: {data.edge_attr.max():.4f}")
            print(f"  Mean: {data.edge_attr.mean():.4f}")
            print(f"  Std: {data.edge_attr.std():.4f}")
        
        # 연결성 확인
        unique_nodes_in_edges = torch.unique(data.edge_index).numel()
        isolated_nodes = data.num_nodes - unique_nodes_in_edges
        print(f"\nConnectivity:")
        print(f"  Connected nodes: {unique_nodes_in_edges}")
        print(f"  Isolated nodes: {isolated_nodes}")

def main():
    """
    메인 실행 함수
    """
    # 데이터 경로 설정
    data_path = "/home/ckswn829/PseudoGraph/selected_data/5mb_13000_1/"
    
    # 그래프 구성기 초기화
    graph_constructor = GraphConstructor(data_path)
    
    try:
        # 그래프 데이터 생성
        graph_data = graph_constructor.create_graph_data(
            threshold=0.01,        # KR 정규화 값 임계값
            top_k_per_node=100,    # 각 노드당 최대 100개 연결
            normalize_edges=True   # 엣지 속성 정규화
        )
        
        # 통계 정보 출력
        graph_constructor.get_graph_statistics()
        
        # 결과 저장
        graph_constructor.save_graph_data("graph_data.pkl")
        
        logger.info("Graph construction completed successfully!")
        
        # 간단한 검증
        print(f"\n=== Validation ===")
        print(f"Graph data type: {type(graph_data)}")
        print(f"Is undirected: {graph_data.is_undirected()}")
        print(f"Has isolated nodes: {graph_data.has_isolated_nodes()}")
        print(f"Has self loops: {graph_data.has_self_loops()}")
        
    except Exception as e:
        logger.error(f"Error during graph construction: {e}")
        raise

if __name__ == "__main__":
    main()
