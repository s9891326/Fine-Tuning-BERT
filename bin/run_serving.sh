#!/usr/bin/env bash
# GPU Monitor : https://github.com/wookayin/gpustat

set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace

# Set magic variables for current file & dir
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__file="${__dir}/$(basename "${BASH_SOURCE[0]}")"
__base="$(basename ${__file} .sh)"
__root="$(cd "$(dirname "${__dir}")" && pwd)" # <-- change this as it depends on your app

# Cygwin
if [[ -x "$(command -v cygpath)" ]]; then
  __dir="$(cygpath -w ${PWD})"
  __file="$(cygpath -w ${__file})"
  __root="$(cygpath -w ${__root})"
fi

#NODE_NAME="${TF_NODE_NAME:-tensorflow_node_0}"
CONTAINER_RAM="4g" # Units : k/m/g
CONTAINER_CPU="4"

serving_models="${__root}/serving/models/"
#serving_models="${__dir}/models/"
#serving_models="./models/"

#image_name="tensorflow/serving:2.2.0-rc2-gpu"
#image_name="tensorflow/serving:2.2.0-gpu"
image_name="tensorflow/serving:latest-gpu"

# Do not add '--privileged'! It will always use all of GPUs.
docker run -d -t --restart=unless-stopped \
--gpus 'device=0' \
--log-opt max-size=256m --log-opt max-file=30 \
--memory="${CONTAINER_RAM}" --cpus="${CONTAINER_CPU}" \
--hostname "${NODE_NAME}" \
--name "${NODE_NAME}" \
-p 8500:8500 \
-v "${serving_models}:/models/" \
"${image_name}" \
--model_config_file="/models/models.config" \
--allow_version_labels_for_unavailable_models=true \
--model_config_file_poll_wait_seconds=2
