import os
import tarfile
import pickle
import numpy as np
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify start token indices.")
    parser.add_argument("--dataset_fname", type=str, required=True,
                        help="Path to the tokenized dataset file (.tar.gz).")
    parser.add_argument("--out_fname", type=str, required=True,
                        help="Path to the file that will contain the serialized Python list of start indices. "
                             "Recommended extension is `.pkl`.")
    args = parser.parse_args()

    dataset_fname = args.dataset_fname
    out_fname = args.out_fname

    base_path = os.path.splitext(os.path.basename(dataset_fname))[0]
    base_path = os.path.splitext(base_path)[0]

    with tarfile.open(dataset_fname, "r:gz") as file:
        file_content_byte = file.extractfile(f"{base_path}/meta.pkl").read()
        meta = pickle.loads(file_content_byte)

        extracted = file.extractfile(f"{base_path}/train.bin")
        train_ids = np.frombuffer(extracted.read(), dtype=np.uint16)

    all_cif_start_indices = []

    for i, id in tqdm(enumerate(train_ids), total=len(train_ids), desc="identifying starts..."):
        token = meta["itos"][id]

        if token == "lattice_\n":
            print("find starting token")
            all_cif_start_indices.append(i)

    print("writing start indices...")
    with open(out_fname, "wb") as f:
        pickle.dump(all_cif_start_indices, f, protocol=pickle.HIGHEST_PROTOCOL)