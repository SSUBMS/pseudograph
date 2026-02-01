import numpy as np
import pandas as pd
import os
import sys
import argparse
import glob

def convert_to_matrix(list_data):
    """
    List format의 HiC 데이터를 adjacency matrix로 변환
    GAT-HiC_generalize.py의 utils.convert_to_matrix 함수와 동일한 기능
    
    Args:
        list_data: numpy array with shape (n, 3) - [row, col, value]
    
    Returns:
        adj_matrix: 2D numpy array - adjacency matrix
    """
    if len(list_data.shape) != 2 or list_data.shape[1] != 3:
        raise ValueError("Input data should have shape (n, 3)")
    
    # 최대 인덱스 찾기
    max_idx = int(max(list_data[:, 0].max(), list_data[:, 1].max())) + 1
    
    # 빈 adjacency matrix 생성
    adj_matrix = np.zeros((max_idx, max_idx))
    
    # 값 채우기
    for row_idx, col_idx, value in list_data:
        adj_matrix[int(row_idx), int(col_idx)] = value
        # HiC 데이터는 대칭이므로 대각선이 아닌 경우 대칭 위치에도 값 설정
        if int(row_idx) != int(col_idx):
            adj_matrix[int(col_idx), int(row_idx)] = value
    
    return adj_matrix

def process_single_file(input_file, output_dir):
    """
    단일 HiC 파일을 처리하여 adjacency matrix 생성
    
    Args:
        input_file: 입력 파일 경로
        output_dir: 출력 디렉토리
    
    Returns:
        bool: 성공 여부
    """
    
    # 파일명에서 확장자 제거하고 출력 파일명 생성
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    if base_name.endswith('.kr'):
        base_name = base_name[:-3]  # .kr 제거
    
    output_file = os.path.join(output_dir, f"{base_name}_matrix.txt")
    stats_file = os.path.join(output_dir, f"{base_name}_stats.txt")
    
    print(f"Processing: {input_file}")
    print(f"Output: {output_file}")
    
    try:
        # 데이터 로드
        list_data = np.loadtxt(input_file)
        print(f"  Loaded {len(list_data)} entries")
        
        # Adjacency matrix 생성
        print(f"  Converting to adjacency matrix...")
        adj_matrix = convert_to_matrix(list_data)
        
        # 대각선을 0으로 설정 (self loops 제거)
        np.fill_diagonal(adj_matrix, 0)
        
        print(f"  Matrix shape: {adj_matrix.shape}")
        print(f"  Non-zero elements: {np.count_nonzero(adj_matrix)}")
        print(f"  Sparsity: {1 - np.count_nonzero(adj_matrix) / adj_matrix.size:.4f}")
        
        # Adjacency matrix 저장
        np.savetxt(output_file, adj_matrix, delimiter='\t', fmt='%.6f')
        print(f"  Saved matrix: {output_file}")
        
        # 통계 정보 저장
        with open(stats_file, 'w') as f:
            f.write(f"File: {base_name}\n")
            f.write(f"Input File: {input_file}\n")
            f.write(f"Matrix Shape: {adj_matrix.shape}\n")
            f.write(f"Total Non-zero Elements: {np.count_nonzero(adj_matrix)}\n")
            f.write(f"Total Elements: {adj_matrix.size}\n")
            f.write(f"Sparsity: {1 - np.count_nonzero(adj_matrix) / adj_matrix.size:.6f}\n")
            f.write(f"Mean Value: {np.mean(adj_matrix):.6f}\n")
            f.write(f"Max Value: {np.max(adj_matrix):.6f}\n")
            f.write(f"Min Value: {np.min(adj_matrix):.6f}\n")
            f.write(f"Standard Deviation: {np.std(adj_matrix):.6f}\n")
        
        print(f"  Saved stats: {stats_file}")
        print(f"  ✓ Success!\n")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {input_file}: {e}\n")
        return False

def process_folder(folder_path):
    """
    지정된 폴더의 모든 .txt 파일을 처리 (단, _matrix.txt, _stats.txt 같은 결과 파일 제외)
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory!")
        return
    
    # .txt 파일들 찾기 (_kr_matrix.txt, _embeddings.txt, _stats.txt 제외)
    pattern = os.path.join(folder_path, "*.txt")
    all_txt_files = glob.glob(pattern)
    target_files = [f for f in all_txt_files if not (
        f.endswith("_matrix.txt") or 
        f.endswith("_embeddings.txt") or 
        f.endswith("_stats.txt") or
        f.endswith("_kr.txt")  # kr 데이터 제외하고 싶으면 이 줄 유지
    )]

    if not target_files:
        print(f"No valid HiC txt files found in '{folder_path}'")
        return
    
    print(f"Found {len(target_files)} HiC .txt files in '{folder_path}'")
    print("=" * 60)
    
    # 각 파일 처리
    success_count = 0
    for txt_file in sorted(target_files):
        if process_single_file(txt_file, folder_path):
            success_count += 1
    
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(target_files)} files")
    
    if success_count < len(target_files):
        print(f"Failed: {len(target_files) - success_count} files")


def main():
    """
    메인 함수 - 명령줄 인자 처리
    """
    
    parser = argparse.ArgumentParser(
        description='Convert HiC _kr.txt files to adjacency matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hic_to_matrix.py 1mb_13000
  python hic_to_matrix.py /path/to/data/5mb_70
  python hic_to_matrix.py . 
        """
    )
    
    parser.add_argument(
        'folder', 
        help='Folder containing _kr.txt files to process'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # 절대 경로로 변환
    folder_path = os.path.abspath(args.folder)
    
    print("HiC Data to Adjacency Matrix Converter")
    print("=" * 60)
    print(f"Target folder: {folder_path}")
    print("=" * 60)
    
    # 폴더 처리
    process_folder(folder_path)

if __name__ == "__main__":
    main()
