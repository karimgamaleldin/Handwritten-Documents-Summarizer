from torchmetrics.text import CharErrorRate
EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
CONTEXT_LENGTH = 15
TEACHER_FORCING_RATIO = 0.95
MINIMIZE_METRIC = True
METRIC_TEACHER_FORCE = 'val_cer'
TOKENIZER_PATH = 'tokenizer/tokenizer.json'
PAD_ID = 0
EOS_ID = 3
DETERMINISTIC = True
NUM_WORKERS = 3
VISION_CONFIGS = {
    'num_layers': 1,
    'd_model': 128,
    'num_heads': 4,
    'dff': 256,
    'maximum_position_encoding': 15,
    'patch_size': 16,
    'patch_stride': 16,
    'patch_padding': 0,
    'in_chans': 1,
    'out_channels': 32,
    'dropout': 0.1
}
DECODER_CONFIGS = {
    'vocab_size': 2048,
    'n_layers': 1,
    'n_heads': 4,
    'd_model': 128,
    'dff': 256,
    'd_inner': 512,
    'seq_len': 15,
    'pad_id': 0,
}
METRIC = CharErrorRate()
