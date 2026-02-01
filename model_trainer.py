import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GATEncoder(nn.Module):
    """
    GAT 기반 그래프 인코더
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_edge_attr: bool = True):
        super(GATEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        
        # 엣지 속성 차원 (1차원 스칼라 값)
        edge_dim = 1 if use_edge_attr else None
        
        # GAT 레이어들
        self.gat_layers = nn.ModuleList()
        
        # 첫 번째 레이어
        self.gat_layers.append(
            GATv2Conv(
                input_dim, 
                hidden_dim, 
                heads=num_heads, 
                dropout=dropout, 
                concat=True,
                edge_dim=edge_dim
            )
        )
        
        # 중간 레이어들
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATv2Conv(
                    hidden_dim * num_heads, 
                    hidden_dim, 
                    heads=num_heads, 
                    dropout=dropout, 
                    concat=True,
                    edge_dim=edge_dim
                )
            )
        
        # 마지막 레이어 (concat=False로 평균)
        self.gat_layers.append(
            GATv2Conv(
                hidden_dim * num_heads, 
                output_dim, 
                heads=num_heads, 
                dropout=dropout, 
                concat=False,
                edge_dim=edge_dim
            )
        )
        
        # 배치 정규화
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
            else:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass
        
        Args:
            x: 노드 특징 (num_nodes, input_dim)
            edge_index: 엣지 인덱스 (2, num_edges)
            edge_attr: 엣지 속성 (num_edges,) - 선택적
            
        Returns:
            torch.Tensor: 노드 임베딩 (num_nodes, output_dim)
        """
        # 엣지 속성 처리
        if self.use_edge_attr and edge_attr is not None:
            # 1차원 벡터를 2차원으로 변환 (num_edges, 1)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
        else:
            edge_attr = None
        
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index, edge_attr=edge_attr)
            
            # 마지막 레이어가 아닌 경우에만 정규화와 활성화 적용
            if i < len(self.gat_layers) - 1:
                x = self.batch_norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

class EdgeDecoder(nn.Module):
    """
    엣지 디코더 - 두 노드 임베딩을 받아서 세포주별 기여도 비율 예측
    """
    def __init__(self, 
                 node_embedding_dim: int, 
                 hidden_dim: int = 128, 
                 output_dim: int = 5,  # 5개 세포주
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(EdgeDecoder, self).__init__()
        
        # 입력 차원 = 두 노드 임베딩을 concat한 크기
        input_dim = node_embedding_dim * 2
        
        layers = []
        
        # 첫 번째 레이어
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # 중간 레이어들
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # 마지막 레이어
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, z_i, z_j):
        """
        Forward pass
        
        Args:
            z_i: 첫 번째 노드 임베딩 (batch_size, embedding_dim)
            z_j: 두 번째 노드 임베딩 (batch_size, embedding_dim)
            
        Returns:
            torch.Tensor: 세포주별 기여도 비율 (batch_size, 5)
        """
        # 두 임베딩 결합
        edge_embedding = torch.cat([z_i, z_j], dim=1)
        
        # MLP를 통과
        logits = self.mlp(edge_embedding)
        
        # Softmax로 비율 변환
        ratios = F.softmax(logits, dim=1)
        
        return ratios

class CellLineRatioPredictionModel(nn.Module):
    """
    전체 모델 - GAT Encoder + Edge Decoder
    """
    def __init__(self, 
                 node_feature_dim: int,
                 gat_hidden_dim: int = 128,
                 gat_output_dim: int = 64,
                 gat_num_heads: int = 4,
                 gat_num_layers: int = 3,
                 decoder_hidden_dim: int = 128,
                 decoder_num_layers: int = 3,
                 dropout: float = 0.1,
                 use_edge_attr: bool = True):
        super(CellLineRatioPredictionModel, self).__init__()
        
        self.encoder = GATEncoder(
            input_dim=node_feature_dim,
            hidden_dim=gat_hidden_dim,
            output_dim=gat_output_dim,
            num_heads=gat_num_heads,
            num_layers=gat_num_layers,
            dropout=dropout,
            use_edge_attr=use_edge_attr
        )
        
        self.decoder = EdgeDecoder(
            node_embedding_dim=gat_output_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=5,  # 5개 세포주
            num_layers=decoder_num_layers,
            dropout=dropout
        )
    
    def forward(self, graph_data, edge_pairs):
        """
        Forward pass
        
        Args:
            graph_data: PyTorch Geometric Data 객체
            edge_pairs: 예측할 엣지 쌍들 (batch_size, 2)
            
        Returns:
            torch.Tensor: 예측된 세포주별 기여도 비율 (batch_size, 5)
        """
        # 1. 그래프 인코딩으로 모든 노드의 임베딩 계산
        node_embeddings = self.encoder(
            graph_data.x, 
            graph_data.edge_index, 
            graph_data.edge_attr
        )
        
        # 2. 엣지 쌍에 해당하는 노드 임베딩 추출
        z_i = node_embeddings[edge_pairs[:, 0]]  # 첫 번째 노드들
        z_j = node_embeddings[edge_pairs[:, 1]]  # 두 번째 노드들
        
        # 3. 엣지 디코딩으로 비율 예측
        predicted_ratios = self.decoder(z_i, z_j)
        
        return predicted_ratios

class ModelTrainer:
    """
    모델 학습 및 평가 클래스
    """
    def __init__(self, 
                 data_path: str,
                 model_config: Dict = None,
                 training_config: Dict = None):
        self.data_path = Path(data_path)
        
        # 기본 설정
        self.model_config = model_config or {
            'gat_hidden_dim': 128,
            'gat_output_dim': 64,
            'gat_num_heads': 4,
            'gat_num_layers': 3,
            'decoder_hidden_dim': 128,
            'decoder_num_layers': 3,
            'dropout': 0.1,
            'use_edge_attr': True
        }
        
        self.training_config = training_config or {
            'batch_size': 512,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'patience': 15,
            'weight_decay': 1e-5
        }
        
        # 데이터 저장용
        self.graph_data = None
        self.ground_truth_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # 모델 및 훈련 관련
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
    
    def load_data(self):
        """
        그래프 데이터와 Ground Truth 데이터를 로드합니다.
        """
        logger.info("Loading data...")
        
        # 그래프 데이터 로드
        with open(self.data_path / "graph_data.pkl", 'rb') as f:
            graph_data_dict = pickle.load(f)
            self.graph_data = graph_data_dict['graph_data']
        
        # Ground Truth 데이터 로드
        with open(self.data_path / "ground_truth_data.pkl", 'rb') as f:
            gt_data_dict = pickle.load(f)
            self.ground_truth_data = gt_data_dict['ground_truth']
        
        logger.info(f"Loaded graph with {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")
        logger.info(f"Loaded {len(self.ground_truth_data)} ground truth entries")
        
        # 데이터 검증
        logger.info(f"Graph data - x shape: {self.graph_data.x.shape}")
        logger.info(f"Graph data - edge_index shape: {self.graph_data.edge_index.shape}")
        if hasattr(self.graph_data, 'edge_attr') and self.graph_data.edge_attr is not None:
            logger.info(f"Graph data - edge_attr shape: {self.graph_data.edge_attr.shape}")
        else:
            logger.warning("No edge attributes found in graph data")
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """
        데이터셋을 훈련/검증/테스트로 분할합니다.
        """
        logger.info("Splitting dataset...")
        
        # 엣지와 비율 데이터 추출
        edges = []
        ratios = []
        
        for data in self.ground_truth_data:
            edges.append(data['edge'])
            ratios.append(data['ratios'])
        
        edges = np.array(edges)
        ratios = np.array(ratios)
        
        # 첫 번째 분할: train + val vs test
        edges_temp, edges_test, ratios_temp, ratios_test = train_test_split(
            edges, ratios, test_size=test_ratio, random_state=42, stratify=None
        )
        
        # 두 번째 분할: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        edges_train, edges_val, ratios_train, ratios_val = train_test_split(
            edges_temp, ratios_temp, test_size=val_size, random_state=42, stratify=None
        )
        
        # 텐서로 변환
        self.train_data = {
            'edges': torch.tensor(edges_train, dtype=torch.long),
            'ratios': torch.tensor(ratios_train, dtype=torch.float)
        }
        
        self.val_data = {
            'edges': torch.tensor(edges_val, dtype=torch.long),
            'ratios': torch.tensor(ratios_val, dtype=torch.float)
        }
        
        self.test_data = {
            'edges': torch.tensor(edges_test, dtype=torch.long),
            'ratios': torch.tensor(ratios_test, dtype=torch.float)
        }
        
        logger.info(f"Dataset split - Train: {len(edges_train)}, Val: {len(edges_val)}, Test: {len(edges_test)}")
    
    def create_model(self):
        """
        모델을 생성하고 디바이스로 이동합니다.
        """
        logger.info("Creating model...")
        
        node_feature_dim = self.graph_data.x.shape[1]
        
        self.model = CellLineRatioPredictionModel(
            node_feature_dim=node_feature_dim,
            **self.model_config
        ).to(self.device)
        
        # 옵티마이저 설정
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    def kl_divergence_loss(self, predicted, target):
        """
        KL Divergence Loss 계산
        """
        # 수치적 안정성을 위해 작은 값 추가
        eps = 1e-8
        predicted = predicted + eps
        target = target + eps
        
        # 정규화 (합이 1이 되도록)
        predicted = predicted / predicted.sum(dim=1, keepdim=True)
        target = target / target.sum(dim=1, keepdim=True)
        
        # KL divergence: KL(target || predicted)
        kl_loss = F.kl_div(
            predicted.log(), 
            target, 
            reduction='batchmean'
        )
        
        return kl_loss
    
    def train_epoch(self, epoch):
        """
        한 에포크 훈련
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 데이터를 배치로 나누기
        batch_size = self.training_config['batch_size']
        dataset = TensorDataset(self.train_data['edges'], self.train_data['ratios'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 그래프 데이터를 디바이스로 이동
        graph_data = self.graph_data.to(self.device)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_edges, batch_ratios in progress_bar:
            batch_edges = batch_edges.to(self.device)
            batch_ratios = batch_ratios.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 예측
            predicted_ratios = self.model(graph_data, batch_edges)
            
            # 손실 계산
            loss = self.kl_divergence_loss(predicted_ratios, batch_ratios)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 진행률 바 업데이트
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """
        검증 세트에서 평가
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        batch_size = self.training_config['batch_size']
        dataset = TensorDataset(self.val_data['edges'], self.val_data['ratios'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        graph_data = self.graph_data.to(self.device)
        
        with torch.no_grad():
            for batch_edges, batch_ratios in dataloader:
                batch_edges = batch_edges.to(self.device)
                batch_ratios = batch_ratios.to(self.device)
                
                predicted_ratios = self.model(graph_data, batch_edges)
                loss = self.kl_divergence_loss(predicted_ratios, batch_ratios)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self):
        """
        전체 훈련 프로세스
        """
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.training_config['patience']
        
        for epoch in range(self.training_config['num_epochs']):
            # 훈련
            train_loss = self.train_epoch(epoch)
            
            # 검증
            val_loss = self.validate()
            
            logger.info(f"Epoch {epoch+1}/{self.training_config['num_epochs']} - "
                       f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                torch.save(self.model.state_dict(), self.data_path / "best_model.pth")
                logger.info(f"New best model saved with val loss: {val_loss:.6f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load(self.data_path / "best_model.pth"))
        logger.info("Training completed!")
    
    def evaluate(self):
        """
        테스트 세트에서 최종 평가
        """
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        batch_size = self.training_config['batch_size']
        dataset = TensorDataset(self.test_data['edges'], self.test_data['ratios'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        graph_data = self.graph_data.to(self.device)
        
        with torch.no_grad():
            for batch_edges, batch_ratios in tqdm(dataloader, desc="Evaluating"):
                batch_edges = batch_edges.to(self.device)
                batch_ratios = batch_ratios.to(self.device)
                
                predicted_ratios = self.model(graph_data, batch_edges)
                loss = self.kl_divergence_loss(predicted_ratios, batch_ratios)
                
                total_loss += loss.item()
                
                all_predictions.append(predicted_ratios.cpu().numpy())
                all_targets.append(batch_ratios.cpu().numpy())
        
        avg_test_loss = total_loss / len(dataloader)
        
        # 예측 결과 합치기
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # 추가 메트릭 계산
        mae = np.mean(np.abs(all_predictions - all_targets))
        mse = np.mean((all_predictions - all_targets) ** 2)
        
        # 세포주별 MAE 계산
        cell_lines = ['GM12878', 'H1Esc', 'HAP1', 'HFF', 'IMR90']
        cell_mae = {}
        for i, cell_line in enumerate(cell_lines):
            cell_mae[cell_line] = np.mean(np.abs(all_predictions[:, i] - all_targets[:, i]))
        
        logger.info(f"Test Results:")
        logger.info(f"  KL Loss: {avg_test_loss:.6f}")
        logger.info(f"  Overall MAE: {mae:.6f}")
        logger.info(f"  Overall MSE: {mse:.6f}")
        logger.info(f"  Cell line MAE:")
        for cell_line, mae_val in cell_mae.items():
            logger.info(f"    {cell_line}: {mae_val:.6f}")
        
        return {
            'test_loss': avg_test_loss,
            'mae': mae,
            'mse': mse,
            'cell_mae': cell_mae,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def plot_training_history(self):
        """
        훈련 히스토리 플롯
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence Loss')
        plt.title('Training History (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.data_path / "training_history.png", dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    메인 실행 함수
    """
    # 데이터 경로
    data_path = "/home/ckswn829/PseudoGraph/selected_data/5mb_13000_1/"
    
    # 모델 설정
    model_config = {
        'gat_hidden_dim': 128,
        'gat_output_dim': 64,
        'gat_num_heads': 4,
        'gat_num_layers': 3,
        'decoder_hidden_dim': 128,
        'decoder_num_layers': 3,
        'dropout': 0.1,
        'use_edge_attr': True
    }
    
    # 훈련 설정
    training_config = {
        'batch_size': 512,
        'learning_rate': 0.001,
        'num_epochs': 50,  # 일단 50 에포크로 줄임
        'patience': 10,
        'weight_decay': 1e-5
    }
    
    # 트레이너 초기화
    trainer = ModelTrainer(data_path, model_config, training_config)
    
    try:
        # 1. 데이터 로드
        trainer.load_data()
        
        # 2. 데이터셋 분할
        trainer.split_dataset()
        
        # 3. 모델 생성
        trainer.create_model()
        
        # 4. 훈련
        trainer.train()
        
        # 5. 평가
        results = trainer.evaluate()
        
        # 6. 훈련 히스토리 플롯
        trainer.plot_training_history()
        
        # 7. 결과 저장
        with open(trainer.data_path / "training_results.pkl", 'wb') as f:
            pickle.dump({
                'results': results,
                'model_config': model_config,
                'training_config': training_config,
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses
            }, f)
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
