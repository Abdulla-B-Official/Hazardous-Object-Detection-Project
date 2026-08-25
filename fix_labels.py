from pathlib import Path

# Label folders
folders = [
    "train/labels",
    "valid/labels",
    "test/labels"
]

# Convert 4 classes into 2 classes
# 0 = Cylinder
# 1 = ShockAbsorber
# 2 = cylinder
# 3 = shock absorber

class_change = {
    0: 0,
    1: 1,
    2: 0,
    3: 1
}

for folder in folders:

    folder_path = Path(folder)

    for file in folder_path.glob("*.txt"):

        new_lines = []

        with open(file, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) >= 5:

                old_class = int(parts[0])

                new_class = class_change[old_class]

                parts[0] = str(new_class)

                new_lines.append(" ".join(parts))

        with open(file, "w") as f:
            f.write("\n".join(new_lines))

print("Label conversion completed!")