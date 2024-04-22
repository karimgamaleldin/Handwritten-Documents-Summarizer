from torchmetrics.text import CharErrorRate
EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 5e-6
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
    'num_layers': 3,
    'd_model': 256,
    'num_heads': 8,
    'dff': 1024,
    'maximum_position_encoding': 100,
    'patch_size': 16,
    'patch_stride': 16,
    'patch_padding': 0,
    'in_chans': 1,
    'out_channels': 32,
    'dropout': 0.1
}
DECODER_CONFIGS = {
    'vocab_size': 512,
    'n_layers': 3,
    'n_heads': 8,
    'd_model': 256,
    'dff': 1024,
    'd_inner': 2048,
    'seq_len': 15,
    'pad_id': 0,
}
VANILLA_DECODER_CONFIGS = {
    'vocab_size': 512,
    'embed_size': 256,
    'dff': 1024,
    'n_layers': 3,
    'num_heads': 8,
    'tie_weights': True
}
METRIC = CharErrorRate()
