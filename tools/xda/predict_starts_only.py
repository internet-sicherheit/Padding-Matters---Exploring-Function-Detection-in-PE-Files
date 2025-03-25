from fairseq.models.roberta import RobertaModel
from pathlib import Path
import sys
import torch
import hashlib


STACK_SIZE = 100
BATCH_SIZE = 512

# TODO: adjust output directory
OUTPUT_DICT = Path("output")


def prepare_batch(batch_data: bytes, model):
    encoded_bytes = model.encode(' '.join(batch_data + ['00'] * (BATCH_SIZE - len(batch_data))))

    return encoded_bytes


def predict_stack(offset: int, data: bytes, model):
    encoded_stack = []
    for i in range(STACK_SIZE):
        current_offset = i * BATCH_SIZE
        if current_offset + BATCH_SIZE < len(data):
            encoded_bytes = prepare_batch(data[current_offset:current_offset + BATCH_SIZE], model)
            encoded_stack.append(encoded_bytes)
        else:
            encoded_bytes = prepare_batch(data[current_offset:], model)
            encoded_stack.append(encoded_bytes)
            break

    stack = torch.stack(encoded_stack, dim=0)
    logprobs = model.predict('funcbound', stack)
    labels = logprobs.argmax(dim=2).view(-1).data
            
    return labels


def process_binary(binary_path: Path):
    print(f"Processing binary {binary_path.name}")
    with open(binary_path, 'rb') as f:
        binary_data = f.read()
        sha256 = hashlib.sha256()
        sha256.update(binary_data)
        sha256 = sha256.hexdigest()
        binary_data = [format(byte, '0x') for byte in binary_data]

    # load the model
    # TODO: adjust to correct model 
    roberta_gpu = RobertaModel.from_pretrained('checkpoints/funcbound', 'checkpoint_best.pt',
                                           'data-bin/funcbound', bpe=None, user_dir='finetune_tasks')
    roberta_gpu.cuda()
    roberta_gpu.eval()

    initial_offset = 0
    binary_data = binary_data[initial_offset:]
    labels = []
    for i in range(0, len(binary_data) - initial_offset, BATCH_SIZE * STACK_SIZE):
        labels += predict_stack(i, binary_data[i:], roberta_gpu)

    labels = torch.stack(labels, dim=0)
    labels.cuda()
    starts_pred = torch.where(labels == 2)[0].tolist()

    output_file = OUTPUT_DICT / f'{sha256}.out'
    output_strs = []
    for start in starts_pred:
        output_strs.append(f'{start}, ')

    with open(output_file, 'w') as out:
        out.write('\n'.join(output_strs))


if __name__ == '__main__':
    # collect the binaries in the given dictionary
    dict_path = Path(sys.argv[1])

    for binary in dict_path.iterdir():
        process_binary(binary)
