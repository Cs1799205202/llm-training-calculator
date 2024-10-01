# nvidia-smi -lgc 900 -i 1
# nvidia-smi -rgc
# nvidia-smi dmon

#!/bin/bash

show_usage() {
    appname=$0
    echo_info "Usage: ${appname} [command], e.g., ${appname} limit 1"
    echo_info "  -- limit [device_id]                limit the gpu memory"
    echo_info "  -- reset                           reset the gpu memory"
    echo_info "  -- monitor                         monitor the gpu memory"
}

export LC_ALL=C
if (( $# == 0 )); then
    echo_warn "Argument cannot be NULL!"
    show_usage
    exit 0
fi

global_choice=${1}
case ${global_choice} in
    "limit")
        local device_id=${2}
        nvidia-smi -lgc 900 -i ${device_id}
        ;;
    "reset")
        nvidia-smi -rgc
        ;;
    "monitor")
        nvidia-smi dmon
        ;;
    *)
        echo "Unrecognized argument!"
        show_usage
esac
