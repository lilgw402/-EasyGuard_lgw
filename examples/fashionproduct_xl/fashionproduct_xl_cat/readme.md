# This is an example of fashionproduct-xl finetune demo.

```bash
# 0. environment preparation
pip3 install https://d.scm.byted.org/api/v2/download/nlp.lib.ptx2_1.0.0.568.tar.gz
pip3 install byted-matxscript==1.8.2
pip3 install http://d.scm.byted.org/api/v2/download/ceph:nlp.tokenizer.py_1.0.0.115.tar.gz
pip3 install https://luban-source.byted.org/repository/scm/search.nlp.libcut_py_matx4_2.3.1.1.tar.gz

download hdfs://harunava/home/byte_magellan_va/user/wangxian/cache/libcut_model_ml_20201229 to /opt/tiger when you train on local machine
or
SET SCM repo "search/nlp/mlcut_files", version "1.0.0.4", path "/opt/tiger/libcut_model_ml_20201229" on merlin or arnold

# 1. can dump default configs as initial config file (in your local machine)
python3 examples/fashoinproduct_xl/fashionproduct_xl_cat/run_train.py --print_config > examples/fashionproduct_xl/fashionproduct_xl_cat/default_config.yaml
# 2. modify the file (remove useless non-config text at the beginning of the document)
vim exampless/fashoinproduct_xl/fashionproduct_xl_cat/default_config.yaml
# 3. load the modified config back
python3 examples/fashoinproduct_xl/fashionproduct_xl_cat/run_train.py --config examples/fashionproduct_xl/fashionproduct_xl_cat/default_config.yaml
or
python3 examples/fashoinproduct_xl/fashionproduct_xl_cat/run_train.py --config hdfs://path/to/your/default_config.yaml 
# 4. customize extra configs manually
python3 examples/fashoinproduct_xl/fashionproduct_xl_cat/run_train.py --config examples/fashionproduct_xl/fashionproduct_xl_cat/default_config.yaml --model.hidden_size=1024
```
