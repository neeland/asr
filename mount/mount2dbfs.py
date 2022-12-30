# Databricks notebook source
# MAGIC %md
# MAGIC https://transform365.blog/2020/06/15/mount-a-blob-storage-in-azure-databricks-only-if-it-does-not-exist-using-python/

# COMMAND ----------

import pyspark
import os

# COMMAND ----------

for mount in dbutils.fs.mounts():
  print (mount.mountPoint)

# COMMAND ----------

storage_account_name = 'callcentrerecordings'
storage_account_access_key = 'KDxWNJYwT00yQd1AhYU0H5vIZjDTHYB4zN4V+Fvhufv/2e/8Dy1MtzWh8AUMU5uF3f7OmT9T5VRV+ASt+XFaoA=='
blob_container = 'recordings'
mount_name = 'asr'

# mount if the mountpoint doesn’t exist
if not any(mount.mountPoint == f'/mnt/{mount_name}' for mount in dbutils.fs.mounts()):
  try:
    dbutils.fs.mount(
    source = f"wasbs://{blob_container}@{storage_account_name}.blob.core.windows.net",
    mount_point = "/mnt/asr/",
    extra_configs = {'fs.azure.account.key.' + storage_account_name + '.blob.core.windows.net': storage_account_access_key}
  )
    
  except Exception as e:
    print("already mounted. Try to unmount first")


# COMMAND ----------

# list files in mounted folder
os.listdir(f'/dbfs/mnt/{mount_name}')

# COMMAND ----------

# unmount uneeded folders
unmount = "FileStore/MountFolder"
dbutils.fs.unmount(f"/mnt/{unmount}")
