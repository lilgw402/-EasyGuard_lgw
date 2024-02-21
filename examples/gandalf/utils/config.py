# coding=utf-8
# Email: jiangxubin@bytedance.com
# Create: 2023/3/9 20:49
from addict import Dict

config = Dict()
# trainer
config.trainer.resume_checkpoint = ""
config.trainer.enable_amp = True
config.trainer.find_unused_parameters = False
config.trainer.summary_interval = 50
config.trainer.use_sync_bn = True
config.trainer.resume_regex = ""
config.trainer.accum_grad_steps = 1
config.trainer.clip_grad_norm = None
config.trainer.eval_output_names = []
config.trainer.gather_val_loss = True
config.trainer.auto_resume = False  # True会加载<hdfs_output_dir>/checkpoints/下的last.pth, 恢复整个训练state（数据load & 模型）
config.trainer.save_ckpt_iterations = -1  # push best ckpt every each iterations
config.trainer.save_last_ckpt_interval = -1
config.trainer.detect_anomaly = False  # for cruise trainer
config.trainer.resume_dataloader = False  # 数据加载是否需要恢复到特定state，注:auto_resume下会强制set True

# data_factory
config.data_factory.batch_size = 8
config.data_factory.batch_size_val = -1  # -1 means not setting，its actual bs for val would be 1/2 of bs in train
config.data_factory.num_workers = 0  # process for preprocess
config.data_factory.num_parallel_reads = 4  # process for fetching data from hdfs
config.data_factory.drop_last = False
config.data_factory.shuffle_files = False  # 可抢占模式下对files进行shuffle，以保证每次都能在不同数据上训练
config.data_factory.multiplex_mix_batch = True  # 混合数据读取下，默认按照一个batch按比例混合数据源
config.data_factory.fast_resume = True
config.data_factory.parquet_cache_on = True  # cruise loader是否开启parquet cache功能，会占用部分磁盘空间


# tester
config.tester.type = "Test"
config.tester.resume_checkpoint = ""
config.tester.override_test_result = True
config.tester.metrics = [
    {
        "type": "AUC",
        "score_key": "output",
        "label_key": "label",
        "show_threshold_details": True,
    }
]
config.tester.from_best_checkpoint = False  # auto iteration used
config.tester.float16 = False
config.tester.only_infer = False  # if true, will not calculate metrics
config.tester.dump_cycle = 100000  # dump json per million rows of test result
config.tester.push_recorddetails_2_hdfs = True  # push test details to your_hdfs_path/record_details
config.tester.push_details_size_limits = 1e6
config.tester.check_resume_checkpoint_exist = True  # 检查resume_checkpoint是否填对

# tracer
config.tracer.type = "Tracer"
config.tracer.resume_checkpoint = ""
config.tracer.trace_format = "onnx"
config.tracer.verify = False
config.tracer.float16 = True
config.tracer.verify_diff_scale = 1e-3
config.tracer.from_best_checkpoint = False
config.tracer.direct_half = False
config.tracer.check_resume_checkpoint_exist = True  # 检查resume_checkpoint是否填对
