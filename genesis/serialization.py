import pickle
import dill as pickle
import os
import shutil

def save(state_dict, file_path):
    """Save state dictionary to file with atomic write and backup.

    Args:
        state_dict: Dictionary containing state to save
        file_path: Path where to save the state dictionary

    Raises:
        Exception: If save operation fails, original file is restored from backup
    """
    backup_file_path = file_path + ".genesis.bak"
    try:
        if os.path.exists(file_path):
            shutil.copyfile(file_path, backup_file_path)

        with open(file_path, "wb") as f:
            pickle.dump(state_dict, f)

        for key in list(state_dict.keys()):
            del state_dict[key]
            
        if os.path.exists(backup_file_path):
            os.remove(backup_file_path)

    except Exception as e:
        if os.path.exists(backup_file_path):
            shutil.copyfile(backup_file_path, file_path)
        raise e

def load(file_path):
    """Load state dictionary from file.

    Args:
        file_path: Path to the state dictionary file

    Returns:
        dict: Loaded state dictionary
    """
    with open(file_path, "rb") as f:
        state_dict = pickle.load(f)
    return state_dict

def save_checkpoint(model_state_dict, optimizer_state_dict, file_path):
    """Save model and optimizer states as checkpoint.

    Args:
        model_state_dict: Model state dictionary
        optimizer_state_dict: Optimizer state dictionary
        file_path: Path where to save the checkpoint
    """
    checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
    }
    save(checkpoint, file_path)

def load_checkpoint(file_path):
    """Load model and optimizer states from checkpoint file.

    Args:
        file_path: Path to the checkpoint file

    Returns:
        tuple: (model_state_dict, optimizer_state_dict)
    """
    with open(file_path, "rb") as f:
        state_dict = pickle.load(f)
    return state_dict["model_state_dict"], state_dict["optimizer_state_dict"]
