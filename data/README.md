# RealWaste data

The image dataset is not committed to Git.

Run the following command after installing the project:

```bash
python scripts/download_data.py --record-checksum
```

`--record-checksum` is a one-time trust action. Confirm that the URL printed in
`PLAN_CODE_REALWASTE.md` is the official UCI Dataset ID 908 download before using it.
Subsequent runs should omit the flag and will verify the pinned SHA-256 value.

Expected extracted layout:

```text
data/raw/RealWaste/<class name>/*.jpg
```

