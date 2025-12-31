import os

path = "dataset"        # thư mục chứa train / valid / test
files = ['train', 'valid', 'test']
TARGET_DATE = "2014-06-03"

output_dir = "filtered_output"
os.makedirs(output_dir, exist_ok=True)

for f in files:
    input_path = os.path.join(path, f)
    output_path = os.path.join(output_dir, f"{f}_2014_06_03.txt")

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                lhs, rel, rhs, timestamp = line.split('\t')
            except ValueError:
                continue  # bỏ dòng lỗi

            if timestamp == TARGET_DATE:
                fout.write(f"{lhs}\t{rel}\t{rhs}\t{timestamp}\n")

    print(f"Done filtering {f}")
