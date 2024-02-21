set -x

# https://www.cyberciti.biz/faq/unix-linux-bash-script-check-if-variable-is-empty/
if [ -z "$MARIANA_OVERRIDE_CRUISE_VERSION" ]
then
    echo "MARIANA_OVERRIDE_CRUISE_VERSION not set, will not update cruise"
else
    echo "MARIANA_OVERRIDE_CRUISE_VERSION set, will update cruise to $MARIANA_OVERRIDE_CRUISE_VERSION"
    cd /opt/tiger;
    rm -rf cruise;
    mkdir -p cruise && cd cruise;
    wget http://luban-source.byted.org/repository/scm/data.aml.cruise_1.0.0.$MARIANA_OVERRIDE_CRUISE_VERSION.tar.gz;
    tar -xf data.aml.cruise*.tar.gz;
fi

if [ -z "$MARIANA_INSTALL_EXTRA_PIP_DEPS" ]
then
    echo "MARIANA_INSTALL_EXTRA_PIP_DEPS not set, will not install extra pip deps in setup_cruise.sh"
else
    echo "MARIANA_INSTALL_EXTRA_PIP_DEPS set, will install extra pip deps in setup_cruise.sh"
    # setup xperf ops
    pip3 install https://luban-source.byted.org/repository/scm/data.aml.lego_ops_th110_cu113_cudnn820_sdist_1.0.0.179.tar.gz

    # fix minor issues from dependencies
    pip3 install -U byted-wandb bytedfeather -i "https://bytedpypi.byted.org/simple"
    pip3 install https://luban-source.byted.org/repository/scm/search.nlp.libcut_py_2.3.0.48.tar.gz --upgrade --force-reinstall
    pip3 install accelerate
    # pip3 install --proxy="" promptsource@git+https://github.com/pku-yao-cheng/promptsource@llm-eval-pipeline
fi
