## generate tfrecords file

```powershell
python build_tfrecord.py --input_dir=/Users/wangminghui/Workspace/tensorflow/classify_dance/data/img --output_dir=test.tfrecords --img_width=120 --img_height=213
```

## read info from tfrecords file

```powershell
python read_tfrecord.py --tfrecord=test.tfrecords
```

