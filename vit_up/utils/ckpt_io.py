import subprocess


def download_ckpt(run_id, step, dest_dir):
    cmd = f"AWS_S3_ADDRESSING_STYLE=path \
		AWS_RETRY_MODE=adaptive \
		AWS_MAX_ATTEMPTS=20 \
		aws s3 cp \
		--region us-ca-2 \
		--endpoint-url https://s3api-us-ca-2.runpod.io \
		s3://10ez88xz0j/logs/nf_dino/{run_id}/checkpoints/epoch-step-{step}.ckpt \
		{dest_dir}/"
    subprocess.run(cmd, shell=True, check=True)


def load_ckpt_configs():
    ckpt_configs = {
        "f05": {
            "run_id": "6ozbejfe",
            "step": 3900,
            "config_path": "../configs/f05_b200.yaml",
        },
        "f04": {
            "run_id": "6fpo61xt",
            "step": 3300,
            "config_path": "../configs/f04_b200.yaml",
        },
        "f01_latest": {
            "run_id": "odxfkvdv",
            "step": 1000,
            "config_path": "../configs/f01_1_b200.yaml",
        },
        "latest": {
            "run_id": "m3opdwdi",
            "step": 13000,
            "config_path": "../configs/16_1_b200_v1.yaml",
        },
        "longest_run": {
            "run_id": "lmq1p3vl",
            "step": 60000,
            "config_path": "../configs/05_1_b200_v3.yaml",
        },
        "10": {
            "run_id": "altbju4q",
            "step": 20000,
            "config_path": "../configs/10_1_b200_v1.yaml",
        },
        "09": {
            "run_id": "hy0t2xfn",
            "step": 1500,
            "config_path": "../configs/09_1_b200_v1.yaml",
        },
        "other_run_ids": ["fb17x9om", "0cdeen72", "lmq1p3vl", "altbju4q", "cyqjnize"],
    }
    return ckpt_configs
