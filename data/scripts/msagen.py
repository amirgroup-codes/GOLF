from evcouplings.utils import read_config_file, write_config_file
from evcouplings.utils.pipeline import execute
import json
import os

config = read_config_file("../EVcouplings/config/sample_config_monomer.txt", preserve_order=True)

cwd = os.getcwd()
with open("params.json", "r") as json_file:
    data = json.load(json_file)
prefix = data["prefix"]
fasta_file = data["fasta_file"]
region = json.loads(data["region"])

config["global"]["prefix"] = os.path.join(cwd, "../results/", prefix)
config["global"]["sequence_file"] = os.path.join(cwd, "../fasta/", fasta_file)
write_config_file(f"config_files/{prefix}.txt", config)

config = read_config_file(f"config_files/{prefix}.txt")
outcfg = execute(**config)

