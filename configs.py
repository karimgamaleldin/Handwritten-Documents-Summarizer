from torchmetrics.text import CharErrorRate
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 5e-6
CONTEXT_LENGTH = 40
TEACHER_FORCING_RATIO = 0.95
MINIMIZE_METRIC = True
METRIC_TEACHER_FORCE = 'val_cer'
TOKENIZER_PATH = 'tokenizer/tokenizer.json'
PAD_ID = 0
EOS_ID = 3
DETERMINISTIC = True
NUM_WORKERS = 3
VISION_CONFIGS = {
    'num_layers': 3,
    'd_model': 512,
    'num_heads': 8,
    'dff': 2048,
    'maximum_position_encoding': 100,
    'patch_size': 16,
    'patch_stride': 16,
    'patch_padding': 0,
    'in_chans': 1,
    'out_channels': 64,
    'dropout': 0.1
}
DECODER_CONFIGS = {
    'vocab_size': 256,
    'n_layers': 3,
    'n_heads': 8,
    'd_model': 512,
    'dff': 2048,
    'd_inner': 4096,
    'seq_len': 15,
    'pad_id': 0,
}
VANILLA_DECODER_CONFIGS = {
    'vocab_size': 256,
    'embed_size': 512,
    'dff': 2048,
    'n_layers': 3,
    'num_heads': 8,
    'tie_weights': True
}
METRIC = CharErrorRate()
