import filecmp
import difflib
import shutil
import os


def copy_diff_files(dcmp):
    diff_files = dcmp.diff_files
    for file in diff_files:
        if os.path.exists("script/megatron4.0/" + file):
            with open(dcmp.left + "/" + file) as f1, open(
                "script/megatron4.0/" + file
            ) as f2:
                diff = difflib.unified_diff(f2.readlines(), f1.readlines())
                print("".join(diff))
        else:
            with open(dcmp.left + "/" + file) as f1, open(
                dcmp.right + "/" + file
            ) as f2:
                diff = difflib.unified_diff(f2.readlines(), f1.readlines())
                print("".join(diff))

        print(f"copying {dcmp.left}/{file} to script/megatron4.0/{file}")
        shutil.copy2(dcmp.left + "/" + file, "script/megatron4.0/" + file)
    for sub_dcmp in dcmp.subdirs.values():
        copy_diff_files(sub_dcmp)


if __name__ == "__main__":
    dcmp = filecmp.dircmp(
        "Megatron",
        "script/Megatron-LM-core_v0.4.0",
        ignore=[
            "__pycache__",
            "datasets",
            "helpers.cpython-310-x86_64-linux-gnu.so",
            "gpu-rank-map.txt",
            "pretrain_gpt_distributed_small.sh",
            "pretrain_gpt_distributed_dataprep_small.sh",
            "benchmark-data-*-.json",
            "build",
            "ngc_models",
            "transformer_engine.py"
        ],
    )
    copy_diff_files(dcmp)
    print("Update finished.")
