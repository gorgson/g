import contextlib
import torch
from torch.cuda.amp import autocast as autocast
import requests
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer,LlavaForConditionalGeneration, LlamaForCausalLM,CLIPImageProcessor, LlamaTokenizerFast, LlavaProcessor,LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import json
from torch import nn
import copy
ignore_index = -100
from tiktoken.load import load_tiktoken_bpe
import dgl
import torch

def generate_subgraphs(g_c, center_node, k,train_idx=None):
    if train_idx==None:
        g = g_c.clone()
        g=g.remove_self_loop()

        # 1. 获取1-hop邻居
        one_hop_neighbors = dgl.sampling.sample_neighbors(g, center_node, k)
        one_hop =[g.ndata["index"][i.item()].item() for i in one_hop_neighbors.edges()[0]]
        one_hop_neighbors =[i.item() for i in one_hop_neighbors.edges()[0]]


        # 2. 获取2-hop邻居
        if len(one_hop_neighbors)!=0:
            two_hop_neighbors = dgl.sampling.sample_neighbors(g, one_hop_neighbors,k//len(one_hop_neighbors))
        else:
            two_hop_neighbors = dgl.sampling.sample_neighbors(g, one_hop_neighbors,k)
        two_hop_neighbors =[i.item() for i in torch.unique(two_hop_neighbors.edges()[0]) if i.item() not in one_hop_neighbors and i.item()!=center_node]
        two_hop=[g.ndata["index"][i].item() for i in two_hop_neighbors[:k//2]]

        
        return one_hop,two_hop
    else:
        g = g_c.clone()
        g=g.remove_self_loop()

        # 1. 获取1-hop邻居
        one_hop_neighbors = dgl.sampling.sample_neighbors(g, center_node, k)
        one_hop =[g.ndata["index"][i.item()].item() for i in one_hop_neighbors.edges()[0] if i in train_idx]
        one_hop_neighbors =[i.item() for i in one_hop_neighbors.edges()[0] if i in train_idx]


        # 2. 获取2-hop邻居
        if len(one_hop_neighbors)!=0:
            two_hop_neighbors = dgl.sampling.sample_neighbors(g, one_hop_neighbors,k//len(one_hop_neighbors))
        else:
            two_hop_neighbors = dgl.sampling.sample_neighbors(g, one_hop_neighbors,k)
        two_hop_neighbors =[i.item() for i in torch.unique(two_hop_neighbors.edges()[0]) if i.item() not in one_hop_neighbors and i.item()!=center_node  and  i.item() in train_idx]
        two_hop=[g.ndata["index"][i].item() for i in two_hop_neighbors[:k//2]]

        
        return one_hop,two_hop

class LLMlp(torch.nn.Module):

    def __init__(
        self,
        prompt,
        
        args,
        g=None,
        train_idx=None,
        text=None,
        graph=None,
        graph_type=None,
        pre_model=None,
        dataset=None
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.pre_model=pre_model
        self.dataset=dataset
        lora_r: int = 8
        lora_alpha: int = 16
        lora_dropout: float = 0.05
        lora_target_modules = [
            "q_proj",
            "v_proj",
        ]
        self.k=args.k
        print('Loading LLamA')
        kwargs = {
            "max_memory": {0: '80GiB',1: '80GiB'},
            "device_map": "auto",
            "revision": "main",
        }
        # self.image_processor = CLIPImageProcessor.from_pretrained(args.llm_model_path)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.processor = AutoProcessor.from_pretrained(args.llm_model_path, revision=kwargs["revision"])
        with open(f"{args.llm_model_path}/special_tokens_map.json", "r") as f:
            self.special_tokens_map = json.load(f)
        self.text=text
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        self.text_proj=nn.Sequential(
            nn.Linear(512, 2048),
            #nn.Sigmoid(),
            #nn.Linear(1024, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).cuda()
        self.img_proj=nn.Sequential(
            nn.Linear(512, 2048),
            #nn.Sigmoid(),
            #nn.Linear(1024, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).cuda()
        model = LlamaForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2",
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLaVA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Fine-tuning LLaVA!")
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLaVA!')
        self.prompt=prompt
        self.word_embedding = self.model.get_input_embeddings()

        self.train_idx=train_idx
        self.g_train=self.train_idx

        self.map=["no","yes"]

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples):
        device=self.model.device
        edges, labels = samples
        edges, labels = edges.to(device), labels.to(device)
        src_features = self.dataset.train_graph.ndata['feat'][edges[:, 0].cpu()]
        dst_features = self.dataset.train_graph.ndata['feat'][edges[:, 1].cpu()]
        src_features=torch.stack([
            self.text_proj(src_features[:,0]),
            self.img_proj(src_features[:,1])
        ],dim=1
        )

        dst_features=torch.stack([
            self.text_proj(dst_features[:,0]),
            self.img_proj(dst_features[:,1])
        ],dim=1
        )
        
        labels=[self.map[j.long().item()] for j in  labels]

        bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        seq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(",",add_special_tokens=False)))
        seqq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(";",add_special_tokens=False)))
        blank_embeds=self.word_embedding(torch.tensor(self.tokenizer.encode(" ",add_special_tokens=False)))
        #bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        questions = self.processor(self.prompt, add_special_tokens=False)

        labels = self.processor(labels, add_special_tokens=False)

        batch_size=edges.shape[0]
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):
            src,dst=edges[i, 0],edges[i, 1]
            one,two=generate_subgraphs(self.g_train, src, self.k)
            o=[]
            t=[]
            for ix,ij in enumerate(one):
                o.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                o.append(seq_embeds.to(self.model.device))
                o.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                o.append(seqq_embeds.to(self.model.device))

      
            for ix,ij in enumerate(two):
                t.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                t.append(seq_embeds.to(self.model.device))
                t.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                t.append(seqq_embeds.to(self.model.device))

            label_input_ids = labels["input_ids"][i]+ [self.tokenizer.eos_token_id]
            raw_text1=self.processor(self.text[src], add_special_tokens=False)
            raw_text2=self.processor(self.text[dst], add_special_tokens=False)
            if o==[]:

                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text1["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    src_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text2["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    dst_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][6] + label_input_ids).to(self.model.device))],
                    dim=0)
            elif t==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text1["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    src_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text2["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    dst_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4]).to(self.model.device)),                
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][6] + label_input_ids).to(self.model.device))],
                    dim=0)
            else:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text1["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    src_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text2["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    dst_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4]).to(self.model.device)),                
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][5]).to(self.model.device)),
                    torch.cat(t,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][6] + label_input_ids).to(self.model.device))],
                    dim=0)

            inputs_embeds = torch.cat([bos_embeds,  inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [ignore_index] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [ignore_index] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss
    def val_forward(self, samples):
        device=self.model.device
        edges, labels = samples
        edges, labels = edges.to(device), labels.to(device)
        src_features = self.dataset.train_graph.ndata['feat'][edges[:, 0].cpu()]
        dst_features = self.dataset.train_graph.ndata['feat'][edges[:, 1].cpu()]
        src_features=torch.stack([
            self.text_proj(src_features[:,0]),
            self.img_proj(src_features[:,1])
        ],dim=1
        )

        dst_features=torch.stack([
            self.text_proj(dst_features[:,0]),
            self.img_proj(dst_features[:,1])
        ],dim=1
        )
        
        
        labels=[self.map[j.long().item()] for j in  labels]

        bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        seq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(",",add_special_tokens=False)))
        seqq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(";",add_special_tokens=False)))
        blank_embeds=self.word_embedding(torch.tensor(self.tokenizer.encode(" ",add_special_tokens=False)))
        #bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        questions = self.processor(self.prompt, add_special_tokens=False)

        labels = self.processor(labels, add_special_tokens=False)

        batch_size=edges.shape[0]
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):
            src,dst=edges[i, 0],edges[i, 1]
            one,two=generate_subgraphs(self.dataset.full_graph, src, self.k,self.dataset.train_idx)
            o=[]
            t=[]
            for ix,ij in enumerate(one):
                o.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                o.append(seq_embeds.to(self.model.device))
                o.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                o.append(seqq_embeds.to(self.model.device))

      
            for ix,ij in enumerate(two):
                t.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                t.append(seq_embeds.to(self.model.device))
                t.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                t.append(seqq_embeds.to(self.model.device))

            label_input_ids = labels["input_ids"][i]+ [self.tokenizer.eos_token_id]
            raw_text1=self.processor(self.text[src], add_special_tokens=False)
            raw_text2=self.processor(self.text[dst], add_special_tokens=False)
            if o==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text1["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    src_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text2["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    dst_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][6] + label_input_ids).to(self.model.device))],
                    dim=0)
            elif t==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text1["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    src_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text2["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    dst_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4]).to(self.model.device)),                
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][6] + label_input_ids).to(self.model.device))],
                    dim=0)
            else:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text1["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    src_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text2["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    dst_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4]).to(self.model.device)),                
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][5]).to(self.model.device)),
                    torch.cat(t,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][6] + label_input_ids).to(self.model.device))],
                    dim=0)

            inputs_embeds = torch.cat([bos_embeds,  inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [ignore_index] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)


        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [ignore_index] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):
        device=self.model.device
        edges, labels = samples
        edges, labels = edges.to(device), labels.to(device)
        src_features = self.dataset.full_graph.ndata['feat'][edges[:, 0].cpu()]
        dst_features = self.dataset.full_graph.ndata['feat'][edges[:, 1].cpu()]
        src_features=torch.stack([
            self.text_proj(src_features[:,0]),
            self.img_proj(src_features[:,1])
        ],dim=1
        )

        dst_features=torch.stack([
            self.text_proj(dst_features[:,0]),
            self.img_proj(dst_features[:,1])
        ],dim=1
        )
        
        labels=[self.map[j.long().item()] for j in  labels]

        bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        seq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(",",add_special_tokens=False)))
        seqq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(";",add_special_tokens=False)))
        blank_embeds=self.word_embedding(torch.tensor(self.tokenizer.encode(" ",add_special_tokens=False)))
        #bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        questions = self.processor(self.prompt, add_special_tokens=False)



        batch_size=edges.shape[0]
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):
            src,dst=edges[i, 0],edges[i, 1]
            one,two=generate_subgraphs(self.dataset.full_graph, src, self.k,self.dataset.train_idx)
            o=[]
            t=[]
            for ix,ij in enumerate(one):
                o.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                o.append(seq_embeds.to(self.model.device))
                o.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                o.append(seqq_embeds.to(self.model.device))

      
            for ix,ij in enumerate(two):
                t.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                t.append(seq_embeds.to(self.model.device))
                t.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                t.append(seqq_embeds.to(self.model.device))

            raw_text1=self.processor(self.text[src], add_special_tokens=False)
            raw_text2=self.processor(self.text[dst], add_special_tokens=False)
            if o==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text1["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    src_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text2["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    dst_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][6]).to(self.model.device))],
                    dim=0)
            elif t==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text1["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    src_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text2["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    dst_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4]).to(self.model.device)),                
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][6]).to(self.model.device))],
                    dim=0)
            else:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text1["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    src_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    self.word_embedding(torch.tensor(raw_text2["input_ids"]).to(self.model.device))[:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    dst_features[i,1].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4]).to(self.model.device)),                
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][5]).to(self.model.device)),
                    torch.cat(t,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][6]).to(self.model.device))],
                    dim=0)

            inputs_embeds = torch.cat([bos_embeds,  inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])







        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]



        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'pred': [p.strip() for p in pred],
                'label': labels}

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
