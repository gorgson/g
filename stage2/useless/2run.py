import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, CLIPImageProcessor, LlamaTokenizerFast, LlavaProcessor
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

# Load the model in half-precision
model = LlavaForConditionalGeneration.from_pretrained(
    "/scratch/ys6310/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

image_processor = CLIPImageProcessor.from_pretrained("/scratch/ys6310/llava-1.5-7b-hf")
tokenizer = LlamaTokenizerFast.from_pretrained("/scratch/ys6310/llava-1.5-7b-hf")
# processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
processor = AutoProcessor.from_pretrained("/scratch/ys6310/llava-1.5-7b-hf")

image_path = "/scratch/ys6310/Mario/dataset/Movies/MoviesImages/0.jpg"
image = Image.open(image_path).convert('RGB')


samples = {
    "id": [1],  # 样本的唯一标识
    "desc": ["Hello"],  # 图片的描述文本
    "question": ["What is in the image?"],  # 关于图片的问题
}
word_embedding = model.get_input_embeddings()

pad_embeds = word_embedding(torch.tensor(tokenizer.pad_token_id).to(0)).unsqueeze(0)
bos_embeds = word_embedding(torch.tensor(tokenizer.bos_token_id).to(0)).unsqueeze(0)

inputs = image_processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs.to(0), vision_feature_layer=-2, vision_feature_select_strategy='default')
image_features = image_features.squeeze(0).to(0)  # 添加 batch 维度

processor.pad_token_id = 0
processor.padding_side = 'left'

model_inputs = processor(samples["desc"][0], add_special_tokens=False)
questions = processor(samples["question"][0], add_special_tokens=False)

special_token_ids_start = processor("USER: ", add_special_tokens=False)
special_token_ids_end = processor("ASSISTANT:", add_special_tokens=False)

special_token_embeds_start = word_embedding(torch.tensor(special_token_ids_start["input_ids"]).squeeze(0).to(0))
special_token_embeds_end = word_embedding(torch.tensor(special_token_ids_end["input_ids"]).squeeze(0).to(0))

batch_inputs_embeds = []
batch_attention_mask = []


input_ids = model_inputs["input_ids"][0] + questions["input_ids"][0]
            

inputs_embeds = word_embedding(torch.tensor(input_ids).to(0))

inputs_embeds = torch.cat([bos_embeds, special_token_embeds_start, image_features,inputs_embeds, special_token_embeds_end], dim=0)

batch_inputs_embeds.append(inputs_embeds)

batch_attention_mask.append([1] * inputs_embeds.shape[0])

max_length = max([x.shape[0] for x in batch_inputs_embeds])

for i in range(1):  # 只有一个样本
    pad_length = max_length - batch_inputs_embeds[i].shape[0]
    batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
    batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(0)
attention_mask = torch.tensor(batch_attention_mask).to(0)

outputs = model.generate(
    inputs_embeds=inputs_embeds,
    max_new_tokens=4096,
    attention_mask=attention_mask,
    # do_sample=True,
    use_cache=True  # IMPORTANT!
)
pred = processor.batch_decode(outputs, skip_special_tokens=True)
print(pred)
