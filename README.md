# Multi-Disc-AVSS

First, Run: 
```
pip install -r requirements.txt
```

The VoxCeleb2 and LRS3-TED datasets are used in this work.



## Train

### Voice Conversion


Generally, run:
```
python train_audio.py --data_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL
```

### Audio-visual Synthesis


Generally, run:
```
python train_audiovisual.py --video_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL --use_256 --load_model LOAD_MODEL_PATH
```

## Test

### Voice Conversion


To convert a wavfile using a trained model, run:
```
python test_audio.py --model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT
```

### Audio-visual Synthesis

Generally run:
```
python test_audiovisual.py --load_model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT --use_256 
```
