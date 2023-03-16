# import pandas as pd
# import numpy as np

# ROWS_PER_FRAME = 543
# def load_relevant_data_subset(pq_path):
#     data_columns = ["x", "y"]
#     data = pd.read_parquet(pq_path, columns=data_columns)

#     print(data.shape)
#     n_frames = int(len(data) / ROWS_PER_FRAME)
#     data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
#     print(data.shape)
#     return data.astype(np.float32)

# load_relevant_data_subset("/home/toefl/K/GISLR/asl-signs/train_landmark_files/2044/635217.parquet")