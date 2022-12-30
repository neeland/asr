# `./asr/mount/*`

# Contents
Each folder has a markdown `.md` file explaining each file in the folder

```
asr
│ 
├── dev            : development
│ 
├── mount          : mounted input files
│    │ 
│    └── README.md : >> you are here <<
│ 
├── prod           : production process
│ 
└── README.md      : project documentation

```

---

## 4️⃣ Mount Azure storage blob using [`mount/mount_blob.py`](mount/mount_blob.py)

---
* Folder contains all components needed to mount machine to Azure blob storage:

* **TODO:** _cleanup neeed; most of these files are no longer needed_

* `mount_blob.py` mounts blob storage using connection config
* `connection.cfg` connection config

```
python mount_blob.py
```

```
mount
├── az_mount_analysis.ipynb
├── connection.cfg
├── mount2dbfs.py
├── mount_README.md
├── mount_blob.py
├── mount_blob_notAzure.py
└── mount_datacrunch.py

```