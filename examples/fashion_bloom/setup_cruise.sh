set -x

cd /opt/tiger;
rm -rf cruise;
mkdir -p cruise && cd cruise;
wget http://luban-source.byted.org/repository/scm/data.aml.cruise_1.0.0.985.tar.gz;
tar -xf data.aml.cruise*.tar.gz;
sudo cp /opt/tiger/cruise/cruise/tools/TORCHRUN /usr/local/bin/TORCHRUN

# setup xperf ops
pip3 install https://luban-source.byted.org/repository/scm/data.aml.lego_ops_th110_cu113_cudnn820_sdist_1.0.0.179.tar.gz

# fix minor issues from dependencies
pip3 install -U byted-wandb bytedfeather -i "https://bytedpypi.byted.org/simple"
pip3 install https://luban-source.byted.org/repository/scm/search.nlp.libcut_py_2.3.0.48.tar.gz --upgrade --force-reinstall

pip3 install accelerate

sudo chmod a+w /tmp
sudo apt update
sudo apt install -y libaio-dev
