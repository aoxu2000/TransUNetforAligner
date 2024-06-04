import os
import scipy.io
import torch
from torch.utils.data import Dataset


class DoseDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_files = self._load_files()

    def _load_files(self):
        data_files = []
        for folder_name in sorted(os.listdir(self.root_dir), key=lambda x: int(x)):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                original_dose_path = os.path.join(folder_path, 'original_dose.mat')
                corrected_dose_path = os.path.join(folder_path, 'corrected_dose.mat')
                if os.path.exists(original_dose_path) and os.path.exists(corrected_dose_path):
                    data_files.append((original_dose_path, corrected_dose_path))
        return data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        original_dose_path, corrected_dose_path = self.data_files[idx]

        # 'D_iter_d_0'
        original_dose = scipy.io.loadmat(original_dose_path)['dose_matrix_0']
        corrected_dose = scipy.io.loadmat(corrected_dose_path)['D_iter_d_0']

        # 将数据转换为PyTorch张量
        original_dose = torch.tensor(original_dose, dtype=torch.float32).unsqueeze(dim=0)
        corrected_dose = torch.tensor(corrected_dose, dtype=torch.float32).unsqueeze(dim=0)

        return original_dose, corrected_dose


if __name__ == '__main__':
    # 示例使用方法
    root_dir = '../data'
    dataset = DoseDataset(root_dir)

    # 测试数据集
    print(f'数据集大小: {len(dataset)}')
    sample_idx = 0
    original_dose, corrected_dose = dataset[sample_idx]
    print(f'原始剂量形状: {original_dose.shape}')
    print(f'修正剂量形状: {corrected_dose.shape}')
