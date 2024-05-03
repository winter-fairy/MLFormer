import torch


def save_model(model, mAP, save_path):
    """
    Save model and the corresponding mAP to save_path
    :param model: model parameters
    :param mAP: mAP of model
    :param save_path: where to save the model
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mAP": mAP
    }
    torch.save(checkpoint, save_path)


def load_model(model, load_path, device):
    """
    Load model parameters
    :param model:
    :param load_path: where to load the model
    :param device: cuda or cpu
    :return: mAP and model
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    max_mAP = checkpoint["mAP"]
    return max_mAP, model