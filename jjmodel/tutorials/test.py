# %%
from pathlib import Path
from collections import Counter
import h5py

folder = Path(
    "/u/akva/JJModel/jjmodel_updated/jjmodel/jjmodel/"
    "input/isochrones/wd/LPCODE"
)

files = sorted(folder.glob("*.h5"))
if not files:
    raise FileNotFoundError(f"No HDF5 files found in {folder}")

models = Counter()

for path in files:
    with h5py.File(path, "r") as h5:
        metadata = dict(h5.attrs)

    print(path.name, metadata)
    models[str(metadata.get("model", "<missing>"))] += 1

print("\nNumber of files:", len(files))
print("Model labels:", dict(models))

# %%
