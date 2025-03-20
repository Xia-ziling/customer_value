# 从kaggle下载所需的数据集
import kagglehub
path = kagglehub.dataset_download("carrie1/ecommerce-data")
print("Path to dataset files:", path)