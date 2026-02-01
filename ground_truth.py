import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroundTruthGenerator:
    def __init__(self, data_path: str):
        """
        Ground Truth 생성기 초기화
        
        Args:
            data_path (str): 데이터 파일들이 있는 경로
        """
        self.data_path = Path(data_path)
        
        # 파일 경로 설정
        self.bulk_file = self.data_path / "bulk_1_matrix.txt"
        self.cell_line_files = {
            'GM12878': self.data_path / "GM12878_bulk_1_23.42%_matrix.txt",
            'H1Esc': self.data_path / "H1Esc_bulk_1_38.03%_matrix.txt",
            'HAP1': self.data_path / "HAP1_bulk_1_19.05%_matrix.txt",
            'HFF': self.data_path / "HFF_bulk_1_17.36%_matrix.txt",
            'IMR90': self.data_path / "IMR90_bulk_1_2.14%_matrix.txt"
        }
        
        # 세포주 순서 (일관성 유지를 위해)
        self.cell_line_order = ['GM12878', 'H1Esc', 'HAP1', 'HFF', 'IMR90']
        
        # 데이터 저장용
        self.bulk_matrix = None
        self.cell_line_matrices = {}
        self.ground_truth_data = []
    
    def load_contact_matrix(self, file_path: Path) -> np.ndarray:
        """
        Contact Matrix 파일을 로드합니다.
        
        Args:
            file_path (Path): 로드할 파일 경로
            
        Returns:
            np.ndarray: Contact Matrix
        """
        try:
            # 텍스트 파일 형태로 저장된 행렬 로드
            matrix = np.loadtxt(file_path)
            logger.info(f"Loaded matrix from {file_path.name}: shape {matrix.shape}")
            return matrix
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def load_all_matrices(self):
        """
        모든 Raw Contact Matrix를 로드합니다.
        """
        logger.info("Loading all contact matrices...")
        
        # Bulk 행렬 로드
        self.bulk_matrix = self.load_contact_matrix(self.bulk_file)
        
        # 각 세포주 행렬 로드
        for cell_line, file_path in self.cell_line_files.items():
            self.cell_line_matrices[cell_line] = self.load_contact_matrix(file_path)
        
        # 모든 행렬의 크기가 같은지 확인
        bulk_shape = self.bulk_matrix.shape
        for cell_line, matrix in self.cell_line_matrices.items():
            if matrix.shape != bulk_shape:
                raise ValueError(f"Matrix size mismatch: {cell_line} has shape {matrix.shape}, "
                               f"but bulk has shape {bulk_shape}")
        
        logger.info(f"All matrices loaded successfully. Matrix size: {bulk_shape}")
    
    def calculate_edge_ratios(self, min_interaction_count: int = 1):
        """
        각 엣지에 대한 세포주별 기여도 비율을 계산합니다.
        
        Args:
            min_interaction_count (int): 최소 상호작용 수 (이보다 적은 상호작용은 제외)
        """
        logger.info("Calculating edge ratios...")
        
        n_bins = self.bulk_matrix.shape[0]
        valid_edges = 0
        skipped_edges = 0
        
        for i in range(n_bins):
            for j in range(i, n_bins):  # 상삼각 행렬만 고려 (대칭성)
                bulk_count = self.bulk_matrix[i, j]
                
                # 최소 상호작용 수 이상인 엣지만 처리
                if bulk_count < min_interaction_count:
                    skipped_edges += 1
                    continue
                
                # 각 세포주의 기여도 계산
                cell_line_counts = []
                for cell_line in self.cell_line_order:
                    count = self.cell_line_matrices[cell_line][i, j]
                    cell_line_counts.append(count)
                
                # 비율 계산 (분모가 0인 경우 방지)
                if bulk_count > 0:
                    ratios = [count / bulk_count for count in cell_line_counts]
                    
                    # 비율의 합이 1에 가까운지 확인 (데이터 일관성 체크)
                    ratio_sum = sum(ratios)
                    if abs(ratio_sum - 1.0) > 0.01:  # 1% 오차 허용
                        logger.warning(f"Edge ({i},{j}): ratio sum = {ratio_sum:.4f} (not close to 1.0)")
                    
                    # Ground Truth 데이터에 추가
                    self.ground_truth_data.append({
                        'edge': (i, j),
                        'bulk_count': bulk_count,
                        'cell_line_counts': cell_line_counts,
                        'ratios': ratios
                    })
                    
                    valid_edges += 1
        
        logger.info(f"Processed {valid_edges} valid edges, skipped {skipped_edges} edges")
        logger.info(f"Total ground truth entries: {len(self.ground_truth_data)}")
    
    def save_ground_truth(self, output_file: str = "ground_truth_data.pkl"):
        """
        Ground Truth 데이터를 파일로 저장합니다.
        
        Args:
            output_file (str): 저장할 파일명
        """
        output_path = self.data_path / output_file
        
        # 저장할 데이터 구성
        save_data = {
            'ground_truth': self.ground_truth_data,
            'cell_line_order': self.cell_line_order,
            'matrix_shape': self.bulk_matrix.shape,
            'metadata': {
                'total_edges': len(self.ground_truth_data),
                'cell_lines': list(self.cell_line_files.keys()),
                'data_path': str(self.data_path)
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Ground truth data saved to {output_path}")
    
    def save_as_csv(self, output_file: str = "ground_truth_data.csv"):
        """
        Ground Truth 데이터를 CSV 형태로도 저장합니다.
        
        Args:
            output_file (str): 저장할 CSV 파일명
        """
        output_path = self.data_path / output_file
        
        # DataFrame 생성
        rows = []
        for data in self.ground_truth_data:
            row = {
                'node_i': data['edge'][0],
                'node_j': data['edge'][1],
                'bulk_count': data['bulk_count']
            }
            
            # 각 세포주의 카운트와 비율 추가
            for idx, cell_line in enumerate(self.cell_line_order):
                row[f'{cell_line}_count'] = data['cell_line_counts'][idx]
                row[f'{cell_line}_ratio'] = data['ratios'][idx]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Ground truth data saved as CSV to {output_path}")
    
    def get_statistics(self):
        """
        Ground Truth 데이터의 통계 정보를 출력합니다.
        """
        if not self.ground_truth_data:
            logger.warning("No ground truth data available. Run calculate_edge_ratios first.")
            return
        
        # 각 세포주별 평균 기여도
        avg_ratios = {cell_line: 0 for cell_line in self.cell_line_order}
        
        for data in self.ground_truth_data:
            for idx, cell_line in enumerate(self.cell_line_order):
                avg_ratios[cell_line] += data['ratios'][idx]
        
        # 평균 계산
        n_edges = len(self.ground_truth_data)
        for cell_line in avg_ratios:
            avg_ratios[cell_line] /= n_edges
        
        print("\n=== Ground Truth Statistics ===")
        print(f"Total edges: {n_edges}")
        print(f"Matrix shape: {self.bulk_matrix.shape}")
        print("\nAverage cell line contributions:")
        for cell_line, avg_ratio in avg_ratios.items():
            print(f"  {cell_line}: {avg_ratio:.4f} ({avg_ratio*100:.2f}%)")
        
        # Bulk count 분포
        bulk_counts = [data['bulk_count'] for data in self.ground_truth_data]
        print(f"\nBulk interaction count statistics:")
        print(f"  Min: {min(bulk_counts)}")
        print(f"  Max: {max(bulk_counts)}")
        print(f"  Mean: {np.mean(bulk_counts):.2f}")
        print(f"  Median: {np.median(bulk_counts):.2f}")

def main():
    """
    메인 실행 함수
    """
    # 데이터 경로 설정
    data_path = "/home/ckswn829/PseudoGraph/selected_data/5mb_13000_1/"
    
    # Ground Truth 생성기 초기화
    gt_generator = GroundTruthGenerator(data_path)
    
    try:
        # 1. 모든 행렬 로드
        gt_generator.load_all_matrices()
        
        # 2. 엣지 비율 계산 (최소 상호작용 수 = 1)
        gt_generator.calculate_edge_ratios(min_interaction_count=1)
        
        # 3. 통계 정보 출력
        gt_generator.get_statistics()
        
        # 4. 결과 저장
        gt_generator.save_ground_truth("ground_truth_data.pkl")
        gt_generator.save_as_csv("ground_truth_data.csv")
        
        logger.info("Ground truth generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ground truth generation: {e}")
        raise

if __name__ == "__main__":
    main()
