import os
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import argparse
import glob

def create_embedding_for_single_file(matrix_file, output_dir):
    """
    단일 matrix 파일에 대해 Node2Vec embedding 생성
    
    Args:
        matrix_file: 입력 matrix 파일 경로
        output_dir: 출력 디렉토리
    
    Returns:
        bool: 성공 여부
    """
    
    # 파일명에서 출력 파일명 생성
    base_name = os.path.splitext(os.path.basename(matrix_file))[0]
    
    # _matrix로 끝나는 경우 _matrix 제거
    if base_name.endswith('_matrix'):
        base_name = base_name[:-7]  # _matrix 제거
    
    embedding_file = os.path.join(output_dir, f"{base_name}_embeddings.txt")
    
    print(f"Processing: {matrix_file}")
    print(f"Output: {embedding_file}")
    
    # 이미 embedding 파일이 존재하는지 확인
    if os.path.isfile(embedding_file):
        print(f"  ✓ Embeddings already exist: {embedding_file}")
        return True
    
    try:
        # 매트릭스 로드
        print(f"  Loading matrix...")
        matrix = np.loadtxt(matrix_file)
        print(f"  Matrix shape: {matrix.shape}")
        
        # 빈 행렬 체크
        if np.count_nonzero(matrix) == 0:
            print(f"  ⚠️  Matrix is empty (all zeros), skipping...")
            return False
        
        # NetworkX 그래프 생성
        print(f"  Creating NetworkX graph...")
        G = nx.from_numpy_array(matrix)
        print(f"  Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
        
        # 연결된 노드가 없는 경우 체크
        if G.number_of_edges() == 0:
            print(f"  ⚠️  Graph has no edges, skipping...")
            return False
        
        # Node2Vec 임베딩 생성
        print(f"  Creating Node2Vec embeddings...")
        node2vec = Node2Vec(G, 
                          dimensions=512, 
                          walk_length=150, 
                          num_walks=50, 
                          p=1.75, 
                          q=0.4, 
                          workers=1, 
                          seed=42)
        
        # 모델 학습
        print(f"  Training model...")
        model = node2vec.fit(window=25, min_count=1, batch_words=4)
        
        # 임베딩 추출
        print(f"  Extracting embeddings...")
        embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
        
        # 임베딩 저장
        np.savetxt(embedding_file, embeddings, fmt='%.6f')
        
        print(f"  ✅ Created embeddings: {embedding_file}")
        print(f"  Stats: shape={embeddings.shape}, mean={np.mean(embeddings):.4f}, std={np.std(embeddings):.4f}")
        print(f"  ✓ Success!\n")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {matrix_file}: {str(e)}\n")
        return False

def process_folder(folder_path):
    """
    지정된 폴더의 모든 *_kr_matrix.txt 파일을 처리
    
    Args:
        folder_path: 처리할 폴더 경로
    """
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory!")
        return
    
    # *_kr_matrix.txt 파일들 찾기
    pattern = os.path.join(folder_path, "*_kr_matrix.txt")
    matrix_files = glob.glob(pattern)
    
    if not matrix_files:
        print(f"No *_kr_matrix.txt files found in '{folder_path}'")
        print(f"Looking for files matching pattern: {pattern}")
        # 디버깅을 위해 폴더의 모든 _matrix.txt 파일 표시
        all_matrix_files = glob.glob(os.path.join(folder_path, "*_matrix.txt"))
        if all_matrix_files:
            print(f"Available *_matrix.txt files in folder:")
            for matrix_file in sorted(all_matrix_files):
                print(f"  - {os.path.basename(matrix_file)}")
        return
    
    print(f"Found {len(matrix_files)} *_kr_matrix.txt files in '{folder_path}'")
    print("Files to process:")
    for matrix_file in sorted(matrix_files):
        print(f"  - {os.path.basename(matrix_file)}")
    print("=" * 60)
    
    # 각 파일 처리
    success_count = 0
    for matrix_file in sorted(matrix_files):
        if create_embedding_for_single_file(matrix_file, folder_path):
            success_count += 1
    
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(matrix_files)} files")
    
    if success_count < len(matrix_files):
        print(f"Failed: {len(matrix_files) - success_count} files")

def main():
    """
    메인 함수 - 명령줄 인자 처리
    """
    
    parser = argparse.ArgumentParser(
        description='Create Node2Vec embeddings for HiC adjacency matrix files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python node2vec_embeddings.py 1mb_13000
  python node2vec_embeddings.py /path/to/data/5mb_70
  python node2vec_embeddings.py . 

This script will process all files matching the pattern *_kr_matrix.txt in the specified folder.
For example, if you have files like:
  - GM12878_bulk_1_kr_matrix.txt
  - H1Esc_bulk_1_kr_matrix.txt
  - HAP1_bulk_1_kr_matrix.txt

The script will generate:
  - GM12878_bulk_1_kr_embeddings.txt
  - H1Esc_bulk_1_kr_embeddings.txt  
  - HAP1_bulk_1_kr_embeddings.txt

Node2Vec Parameters:
  - dimensions: 512
  - walk_length: 150
  - num_walks: 50
  - p: 1.75
  - q: 0.4
  - window: 25
        """
    )
    
    parser.add_argument(
        'folder', 
        help='Folder containing *_kr_matrix.txt files to process'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # 절대 경로로 변환
    folder_path = os.path.abspath(args.folder)
    
    print("Node2Vec Embedding Generator for HiC Data")
    print("=" * 60)
    print(f"NetworkX version: {nx.__version__}")
    print(f"Target folder: {folder_path}")
    print(f"Looking for files matching: *_kr_matrix.txt")
    print("=" * 60)
    
    # 폴더 처리
    process_folder(folder_path)

if __name__ == "__main__":
    main()
