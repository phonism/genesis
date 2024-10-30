import pickle
import dill as pickle
import os
import shutil

def save(state_dict, file_path):
    backup_file_path = file_path + ".genesis.bak"
    try:
        if os.path.exists(file_path):
            shutil.copyfile(file_path, backup_file_path)

        with open(file_path, "wb") as f:
            pickle.dump(state_dict, f)
            
        if os.path.exists(backup_file_path):
            os.remove(backup_file_path)

    except Exception as e:
        if os.path.exists(backup_file_path):
            shutil.copyfile(backup_file_path, file_path)
        raise e

def load(file_path):
    with open(file_path, "rb") as f:
        state_dict = pickle.load(f)
    return state_dict

def save_checkpoint(model_state_dict, optimizer_state_dict, file_path):
    checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
    }
    save(checkpoint, file_path)

def load_checkpoint(file_path):
    with open(file_path, "rb") as f:
        state_dict = pickle.load(f)
    return state_dict["model_state_dict"], state_dict["optimizer_state_dict"]
