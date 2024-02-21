# This is an example of fashionbertv2 pretrain demo.

```bash
# 1. can dump default configs as initial config file (in your local machine)
python3 run_pretrain.py --print_config > default_config.yaml
# 2. modify the file
cat default_config.yaml
# 3. load the modified config back
python3 run_pretrain.py --config default_config.yaml
or
python3 run_pretrain.py --config hdfs://path/to/your/default_config.yaml 
# 4. customize extra configs manually
python3 run_pretrain.py --config default_config.yaml --model.hidden_size=1024
```

**run on single cpu/gpu:**
- `python run_pretrain.py`

**run on multi gpus:**
- local machine: `/path/to/your/local/EasyGuard/tools/TORCHRUN run_pretrain.py`
- arnold: `/opt/tiger/EasyGuard/tools/TORCHRUN run_pretrain.py`