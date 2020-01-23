from detector import train
import os

train("./train_dir", model_save_path="trained_model.clf", n_neighbors=2,delete_unfit_files=True)