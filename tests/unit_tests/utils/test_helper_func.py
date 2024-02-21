import os
import shutil

from easyguard.utils.helper_func import download_model_weights_from_tos


def test_download_model_weights_from_tos():
    dst_dir = "./tmp"
    os.makedirs(dst_dir, exist_ok=True)
    model_name = "bert_base_uncased"
    shutil.rmtree(dst_dir)
    model_path = download_model_weights_from_tos(
        model_name=model_name,
        dst_dir=dst_dir,
    )
    assert os.path.exists(os.path.join(model_path, "config.json"))
    assert os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
    assert os.path.exists(os.path.join(model_path, "tokenizer_config.json"))
    assert os.path.exists(os.path.join(model_path, "vocab.txt"))
