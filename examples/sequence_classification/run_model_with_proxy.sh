# export http_proxy=10.20.47.147:3128
# export https_proxy=10.20.47.147:3128
# export no_proxy=code.byted.org

export http_proxy=http://10.20.47.147:3128 https_proxy=https://10.20.47.147:3128 export no_proxy=code.byted.org
# export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 export no_proxy=byted.org

python3 run_model.py --config config_roberta.yaml

unset http_proxy
unset https_proxy
