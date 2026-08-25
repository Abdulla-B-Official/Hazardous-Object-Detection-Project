from pathlib import Path

folders = [
    "train/labels",
    "valid/labels",
    "test/labels"
]

for folder in folders:

    print("\nChecking:", folder)

    class_ids = set()
    total_files = 0

    for file in Path(folder).glob("*.txt"):

        total_files += 1

        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_ids.add(class_id)

    print("Label files:", total_files)
    print("Class IDs found:", sorted(class_ids))

print("\nCheck completed!")