import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
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
import time
import json
from torch import nn
import copy
ignore_index = -100
from tiktoken.load import load_tiktoken_bpe
import dgl
import copy
import torch.nn.functional as F
from collections import defaultdict
# DQN相关：定义用于存储转换的tuple结构
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
def create_proj(input_dim, num_layer):
    layers = []
    current_dim = input_dim  # 初始化输入维度

    # 添加前 num_layer - 1 个 Linear 层
    for _ in range(num_layer - 1):
        layers.append(nn.Linear(current_dim, 2048))
        layers.append(nn.Sigmoid())  # 每两个 Linear 层后添加一个 Sigmoid
        current_dim = 2048  # 线性层的输出维度是 2048

    # 最后一层 Linear 层，不需要 Sigmoid
    layers.append(nn.Linear(current_dim, 4096))

    return nn.Sequential(*layers).cuda()
def personalized_pagerank(g, alpha=0.85, max_iter=100):
    # Initialize PageRank scores
    pr_scores = torch.ones(g.num_nodes()) / g.num_nodes()
    
    # Out-degree normalization
    out_degrees = g.out_degrees().float()
    out_degrees[out_degrees == 0] = 1  # Prevent division by zero
    
    for _ in range(max_iter):
        # Normalize and propagate scores
        pr_scores = (1 - alpha) / g.num_nodes() + alpha * torch.matmul(g.adjacency_matrix().to_dense(), pr_scores / out_degrees)
        
    return pr_scores
# DQN新增：经验回放缓冲区
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    def pop(self):
        if self.memory:
            return self.memory.pop()
        else:
            return None
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN新增：Q网络定义
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return F.softmax(self.layers(x), dim=-1)

class LLM(torch.nn.Module):
    def __init__(self, prompt, args, g=None, train_idx=None, text=None, map=None, graph=None, graph_type=None, pre_model=None):
        super().__init__()
        # 原有初始化逻辑...
        self.k = args.k
        # DQN动作空间及网络初始化
        self.actions = ['text', 'image', 'pagerank', 'concatenated','stop']
        node_feat_dim = 1024  # Mario生成特征维度
        self.max_neighbors = args.k
        state_dim = node_feat_dim + 1 + 3
        action_dim = len(self.actions)
        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1
        self.optimizer_dqn = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.last_forward_loss = None
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.pre_model=pre_model

        lora_r: int = 8
        lora_alpha: int = 16
        lora_dropout: float = 0.05
        lora_target_modules = [
            "q_proj",
            "v_proj",
        ]
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
        self.map=map
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        self.text_proj=create_proj(512, args.nl)
        self.img_proj=create_proj(512, args.nl)
        model = LlamaForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2",
            **kwargs
        )
        self.stra=args.strategy
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
        self.g=g
        self.train_idx=train_idx
        self.g_train=dgl.node_subgraph(g, self.train_idx)
        self.train_map={j.item():i for i,j in enumerate(self.g_train.ndata["index"])}
        self.k=args.k
        self.pr_scores = personalized_pagerank(self.g_train, alpha=0.85, max_iter=100)
        self.queues=defaultdict(list)
        self.neighbor_cache=defaultdict(list)
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(len(self.actions))
        with torch.no_grad():
            q = self.policy_net(state)
            return q.argmax().item()

    def train_dqn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        next_states = batch.next_state
        mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
        next_batch = torch.cat([s for s in next_states if s is not None]) if mask.any() else torch.empty(0)

        current_q = self.policy_net(state_batch).gather(1, action_batch)
        next_q = torch.zeros(self.batch_size, 1)
        if mask.any():
            next_q[mask] = self.target_net(next_batch).max(1)[0].unsqueeze(1).detach()
        expected_q = reward_batch + self.gamma * next_q
        loss = F.mse_loss(current_q, expected_q)
        self.optimizer_dqn.zero_grad()
        loss.backward()
        self.optimizer_dqn.step()
        self.target_net.load_state_dict(self.policy_net.state_dict())
    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


    def _topk_by_similarity(self, center_node, neighbors, k, fl,modality='text'):
        text_c, img_c = self.feature[center_node, 0], self.feature[center_node, 1]
    
        sims = []
        for n,hop in zip(neighbors,fl):
            text_n, img_n = self.feature[n, 0], self.feature[n, 1]
            if modality == 'text':
                
                sim = F.cosine_similarity(text_c.unsqueeze(0), text_n.unsqueeze(0))

            elif modality == 'image':
                sim = F.cosine_similarity(img_c.unsqueeze(0), img_n.unsqueeze(0))
            elif modality == 'concatenated':
                concat_c = torch.cat((text_c, img_c), dim=-1)
                concat_n = torch.cat((text_n, img_n), dim=-1)
                sim = F.cosine_similarity(concat_c.unsqueeze(0), concat_n.unsqueeze(0))
            sims.append((n, sim.item(),hop))
        sims.sort(key=lambda x: x[1], reverse=True)
        q=deque()
        for n, sim,hop in sims:
            if len(q) < k:
                q.append((n,hop))
            else:
                break
        return q

    def _pagerank_subsampling(self, g, neighbors, k, fl,alpha=0.85, max_iter=100):
        # Compute personalized PageRank scores

        
        # Extract scores for the neighbors
        neighbor_scores = self.pr_scores[neighbors]
        k = min(k, len(neighbor_scores))
        # Sort neighbors by their PageRank scores
        _, topk_indices = torch.topk(neighbor_scores, k, largest=True)
        q=deque()
        for i in topk_indices:
            if len(q) < k:
                q.append((neighbors[i],fl[i]))
            else:
                break
        
        return q

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


    def generate_subgraphs(self, g_c, center_node, train=False):
        # 1. 克隆图
        g = g_c.clone()

        # 2. 邻居采样缓存
        if not train and center_node in self.neighbor_cache:
            one_hop, two_hop = self.neighbor_cache[center_node]
        else:
            one_edges = dgl.sampling.sample_neighbors(g, center_node, -1)
            one_hop = list(set(one_edges.edges()[0].tolist()))
            two_edges = dgl.sampling.sample_neighbors(g, one_hop, -1)
            two_hop = list(set(two_edges.edges()[0].tolist()) - set(one_hop) - {center_node})
            if not train:
                self.neighbor_cache[center_node] = (one_hop, two_hop)

        # 3. 队列缓存与计算
        if not train and center_node in self.queue_cache:
            qt, qi, qc, qp, init_lens = self.queue_cache[center_node]
            queue_text = deque(qt.copy())
            queue_image = deque(qi.copy())
            queue_cat = deque(qc.copy())
            queue_pr = deque(qp.copy())
        else:
            two_hop_list = one_hop + two_hop
            fl = [1] * len(one_hop) + [2] * len(two_hop)
            queue_text = self._topk_by_similarity(center_node, two_hop_list, self.max_neighbors, fl, modality='text')
            queue_image = self._topk_by_similarity(center_node, two_hop_list, self.max_neighbors, fl, modality='image')
            queue_cat = self._topk_by_similarity(center_node, two_hop_list, self.max_neighbors, fl, modality='concatenated')
            queue_pr = self._pagerank_subsampling(g, two_hop_list, self.max_neighbors, fl)
            init_lens = (len(queue_text), len(queue_image), len(queue_pr))
            if not train:
                self.queue_cache[center_node] = (
                    list(queue_text), list(queue_image), list(queue_cat), list(queue_pr), init_lens
                )

        # 4. DQN 采样
        center_feat = torch.cat([self.feature[center_node, 0], self.feature[center_node, 1]], dim=-1)
        two_hop_, one_hop_, cnt = [], one_hop[:5], 0
        count = torch.tensor([0.0])
        ratios = torch.tensor([1.0, 1.0, 1.0])
        state = torch.cat([center_feat, count, ratios]).unsqueeze(0)
        self.buffer = None
        while True:
            a = self.select_action(state)
            act = self.actions[a]
            if act == 'stop' or cnt >= self.max_neighbors:
                self.buffer = self.memory.pop()
                break
            if act == 'text' and queue_text:
                node = queue_text.popleft()
            elif act == 'image' and queue_image:
                node = queue_image.popleft()
            elif act == 'pagerank' and queue_pr:
                node = queue_pr.popleft()
            elif act == 'concatenated' and queue_cat:
                node = queue_cat.popleft()
            else:
                continue
            if node[1] == 1 and node[0] not in one_hop_:
                one_hop_.append(node[0])
                cnt += 1
            elif node[1] == 2 and node[0] not in two_hop_:
                two_hop_.append(node[0])
                cnt += 1
            count = torch.tensor([cnt / self.max_neighbors])
            ratios = torch.tensor([
                len(queue_text) / init_lens[0],
                len(queue_image) / init_lens[1],
                len(queue_pr) / init_lens[2]
            ])
            state = torch.cat([center_feat, count, ratios]).unsqueeze(0)

        # 5. Collect results
        one_info = [(g.ndata['index'][n].item(), g.ndata['label'][n]) for n in one_hop_]
        two_info = [(g.ndata['index'][n].item(), g.ndata['label'][n]) for n in two_hop_]

        return one_info, two_info


    def forward(self, samples):
        device=self.model.device
        input_nodes, output_nodes, blocks=samples
        x = copy.deepcopy(blocks[0].srcdata)
        idx=blocks[-1].dstdata["index"].cpu()
        src_nodes, dst_nodes = blocks[-1].edges()
        blocks=[i.to(self.model.device) for i in blocks]
        num_node=blocks[-1].num_dst_nodes()

        text_embedding, image_embedding = self.feature[idx,0].to(device),self.feature[idx,1].to(device)
        text_embedding=self.text_proj(text_embedding)
        image_embedding=self.img_proj(image_embedding)
        text_embedding, image_embedding = text_embedding[:num_node,], image_embedding[:num_node,]
        labels=[self.map[j.long().item()] for j in  blocks[-1].dstdata["label"]]

        bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        seq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(",",add_special_tokens=False)))
        seqq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(";",add_special_tokens=False)))
        blank_embeds=self.word_embedding(torch.tensor(self.tokenizer.encode(" ",add_special_tokens=False)))
        #bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        questions = self.processor(self.prompt, add_special_tokens=False)

        labels = self.processor(labels, add_special_tokens=False)

        batch_size = num_node
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):

            one,two=self.generate_subgraphs(self.g_train, self.train_map[idx[i].item()],train=True)
            o=[]
            t=[]
            for ix,(ij,labe) in enumerate(one):
                o.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                o.append(seq_embeds.to(self.model.device))
                o.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                o.append(seq_embeds.to(self.model.device))
                o.append(self.word_embedding(
                    torch.tensor(
                        self.processor(
                            self.map[labe.long().item()],add_special_tokens=False)["input_ids"]
                            )).to(self.model.device))
                o.append(seqq_embeds.to(self.model.device))
            for ix,(ij,labe) in enumerate(two):
                t.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                t.append(seq_embeds.to(self.model.device))
                t.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                t.append(seq_embeds.to(self.model.device))
                t.append(self.word_embedding(torch.tensor(self.processor(self.map[labe.long().item()],add_special_tokens=False)["input_ids"])).to(self.model.device))
                t.append(seqq_embeds.to(self.model.device))
            label_input_ids = labels["input_ids"][i]+ [self.tokenizer.eos_token_id]
            raw_text=self.processor(self.text[x["index"][i]], add_special_tokens=False)
            if self.max_txt_len==1:
                texte=self.word_embedding(torch.tensor(raw_text["input_ids"]).to(self.model.device)).mean(dim=0,keepdim=True)
            else:
                texte=self.word_embedding(torch.tensor(raw_text["input_ids"]).to(self.model.device))[:self.max_txt_len]

            if o==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    texte,

                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    image_embedding[i].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4] + label_input_ids).to(self.model.device))],
                    dim=0)                    

            elif t==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    texte,

                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    image_embedding[i].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4] + label_input_ids).to(self.model.device))],
                    dim=0)

            else:

                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    texte,
                    #self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    #text_embedding[i][:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    image_embedding[i].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    torch.cat(t,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4] + label_input_ids).to(self.model.device))],
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
            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
            logits = out.logits              # 取出模型输出的 logits (batch, seq_len, vocab_size)
            labels = label_input_ids                 # 真实标签，形状 (batch, seq_len)
            # preds = logits.argmax(dim=-1)   # 每个位置的预测 token id (batch, seq_len)
            # mask = labels != ignore_index   # 形状 (batch, seq_len)，bool 张量

            # # 计算正确预测数量（只计算 mask 为 True 的位置）
            # correct = (preds == labels) & mask
            # accuracy = correct.sum().float() / mask.sum().float()
            # 原始（未对齐）accuracy
            # orig_preds = logits.argmax(dim=-1)   # (B, L)
            # orig_mask  = labels != ignore_index
            # orig_acc   = (orig_preds == labels)[orig_mask].float().mean()
            # print(orig_preds[orig_mask].shape)
            # 对齐后 accuracy
            shifted_preds = logits[:, :-1, :].argmax(dim=-1)
            shifted_gold  = labels[:, 1:]
            shifted_mask  = shifted_gold != ignore_index
            accuracy  = (shifted_preds == shifted_gold)[shifted_mask].float().mean()
        
            if self.buffer!=None and accuracy>0.2:
                print(accuracy)
                print(self.buffer.action)
                self.memory.push(self.buffer.state, self.buffer.action, torch.tensor([accuracy]), self.buffer.next_state)
        
                self.train_dqn()
  
        return out.loss
 
    # def forward(self, samples):
    #     device = self.model.device
    #     input_nodes, output_nodes, blocks = samples
    #     blocks = [b.to(device) for b in blocks]
    #     x = copy.deepcopy(blocks[0].srcdata)
    #     idx = blocks[-1].dstdata['index'].cpu()
    #     num_node = blocks[-1].num_dst_nodes()
    #     text_e, img_e = self.feature[idx,0].to(device), self.feature[idx,1].to(device)
    #     text_e = self.text_proj(text_e); img_e = self.img_proj(img_e)
    #     text_e, img_e = text_e[:num_node], img_e[:num_node]
    #     bos = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)
    #     pad = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
    #     seq = self.word_embedding(torch.tensor(self.tokenizer.encode(',',add_special_tokens=False)))
    #     seqq = self.word_embedding(torch.tensor(self.tokenizer.encode(';',add_special_tokens=False)))
    #     questions = self.processor(self.prompt, add_special_tokens=False)
    #     labels = self.processor([self.map[i.item()] for i in blocks[-1].dstdata['label']], add_special_tokens=False)
    #     batch_inputs, batch_masks, batch_labels = [], [], []
    #     for i in range(num_node):
    #         one, two = self.generate_subgraphs(self.g_train, self.train_map[idx[i].item()],train=True)
    #         # 以下保持原有拼接逻辑，仅将strategy采样替换为DQN采样结果
    #         o, t = [], []
    #         for (nid, lab) in one:
    #             o.extend([self.text_proj(self.feature[nid,[0]].to(device)), seq, self.img_proj(self.feature[nid,[1]].to(device)), seq,
    #                       self.word_embedding(torch.tensor(self.processor(self.map[lab.item()],add_special_tokens=False)['input_ids'])), seqq])
    #         for (nid, lab) in two:
    #             t.extend([self.text_proj(self.feature[nid,[0]].to(device)), seq, self.img_proj(self.feature[nid,[1]].to(device)), seq,
    #                       self.word_embedding(torch.tensor(self.processor(self.map[lab.item()],add_special_tokens=False)['input_ids'])), seqq])

    #         raw_text=self.processor(self.text[x["index"][i]], add_special_tokens=False)
    #         if self.max_txt_len==1:
    #             txt_emb=self.word_embedding(torch.tensor(raw_text["input_ids"]).to(self.model.device)).mean(dim=0,keepdim=True)
    #         else:
    #             txt_emb=self.word_embedding(torch.tensor(raw_text["input_ids"]).to(self.model.device))[:self.max_txt_len]
    #         inputs = [bos, txt_emb, self.word_embedding(torch.tensor(questions['input_ids'][1]).to(device)),
    #                   img_e[i].unsqueeze(0)]
    #         if o: inputs += [self.word_embedding(torch.tensor(questions['input_ids'][2]).to(device)), torch.cat(o,0)]
    #         if t: inputs += [self.word_embedding(torch.tensor(questions['input_ids'][3]).to(device)), torch.cat(t,0)]
    #         inputs += [self.word_embedding(torch.tensor(questions['input_ids'][4] + labels['input_ids'][i] + [self.tokenizer.eos_token_id]).to(device))]
    #         emb = torch.cat(inputs,0)
    #         batch_inputs.append(torch.cat([pad.repeat(max(0, self.max_new_tokens-emb.shape[0]),1), emb],0))
    #         batch_masks.append([1]*emb.shape[0])
    #         batch_labels.append([ignore_index]*(emb.shape[0]-len(labels['input_ids'][i])) + labels['input_ids'][i])
    #     max_len = max(len(x) for x in batch_inputs)
    #     for i in range(len(batch_inputs)):
    #         p = max_len - batch_inputs[i].shape[0]
    #         if p>0:
    #             batch_inputs[i] = torch.cat([pad.repeat(p,1), batch_inputs[i]],0)
    #             batch_masks[i] = [0]*p + batch_masks[i]
    #             batch_labels[i] = [ignore_index]*p + batch_labels[i]
    #     inputs = torch.stack(batch_inputs,0).to(device)
    #     masks  = torch.tensor(batch_masks).to(device)
    #     lbls   = torch.tensor(batch_labels).to(device)
    #     with self.maybe_autocast():
    #         out = self.model(inputs_embeds=inputs, attention_mask=masks, return_dict=True, labels=lbls)
    #     # 存储当前batch的训练loss，用于DQN的reward计算
    #         logits = out.logits                             # [B, L, V]
    #         preds = logits.argmax(dim=-1)           # [B, L]
    #         # 2. 生成有效位置 mask（忽略 ignore_index）
    #         valid_mask = lbls != ignore_index               # [B, L]
    #         # 3. 统计正确预测数和有效总数
    #         num_correct = ((preds == lbls) & valid_mask).sum().item()
    #         num_total   = valid_mask.sum().item()
    #         accuracy = num_correct / num_total if num_total > 0 else 0.0
    #         if self.buffer!=None and accuracy>0.2:
    #             print(accuracy)
    #             print(self.buffer.action)
    #             self.memory.push(self.buffer.state, self.buffer.action, torch.tensor([accuracy]), self.buffer.next_state)
        
    #             self.train_dqn()
    #     return out.loss
    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
    def val_forward(self, samples):
        device=self.model.device
        input_nodes, output_nodes, blocks=samples
        x = copy.deepcopy(blocks[0].srcdata)
        idx=blocks[-1].dstdata["index"].cpu()
        src_nodes, dst_nodes = blocks[-1].edges()
        blocks=[i.to(self.model.device) for i in blocks]
        num_node=blocks[-1].num_dst_nodes()

        text_embedding, image_embedding = self.feature[idx,0],self.feature[idx,1]

        text_embedding=self.text_proj(text_embedding.to(device))
        image_embedding=self.img_proj(image_embedding.to(device))
        text_embedding, image_embedding = text_embedding[:num_node,].to(device), image_embedding[:num_node,].to(device)
        labels=[self.map[j.long().item()] for j in  blocks[-1].dstdata["label"]]

        bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        seq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(",",add_special_tokens=False)))
        seqq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(";",add_special_tokens=False)))
        blank_embeds=self.word_embedding(torch.tensor(self.tokenizer.encode(" ",add_special_tokens=False)))
        #bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        questions = self.processor(self.prompt, add_special_tokens=False)
        labels = self.processor(labels, add_special_tokens=False)

        batch_size = num_node
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):
            self.g_val=dgl.node_subgraph(self.g, self.train_idx.tolist()+[x["index"][i].item()])
            indices=torch.nonzero(self.g_val.ndata["index"]==x["index"][i].item())[0]
            one,two=self.generate_subgraphs(self.g_val, indices.item())
            o=[]
            t=[]
            for ix,(ij,labe) in enumerate(one):
                o.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                o.append(seq_embeds.to(self.model.device))
                o.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                o.append(seq_embeds.to(self.model.device))
                o.append(self.word_embedding(
                    torch.tensor(
                        self.processor(
                            self.map[labe.long().item()],add_special_tokens=False)["input_ids"]
                            )).to(self.model.device))
                o.append(seqq_embeds.to(self.model.device))
            for ix,(ij,labe) in enumerate(two):
                t.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                t.append(seq_embeds.to(self.model.device))
                t.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                t.append(seq_embeds.to(self.model.device))
                t.append(self.word_embedding(torch.tensor(self.processor(self.map[labe.long().item()],add_special_tokens=False)["input_ids"])).to(self.model.device))
                t.append(seqq_embeds.to(self.model.device))
            label_input_ids = labels["input_ids"][i]+ [self.tokenizer.eos_token_id]
            raw_text=self.processor(self.text[x["index"][i]], add_special_tokens=False)
            if self.max_txt_len==1:
                texte=self.word_embedding(torch.tensor(raw_text["input_ids"]).to(self.model.device)).mean(dim=0,keepdim=True)
            else:
                texte=self.word_embedding(torch.tensor(raw_text["input_ids"]).to(self.model.device))[:self.max_txt_len]
            if o==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    texte,

                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    image_embedding[i].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4] + label_input_ids).to(self.model.device))],
                    dim=0)
            elif t==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    texte,
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    image_embedding[i].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4] + label_input_ids).to(self.model.device))],
                    dim=0)
            else:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    texte,
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    image_embedding[i].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    torch.cat(t,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4] + label_input_ids).to(self.model.device))],
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
        input_nodes, output_nodes, blocks=samples
        x = copy.deepcopy(blocks[0].srcdata)
        idx=blocks[-1].dstdata[dgl.NID].cpu()
        src_nodes, dst_nodes = blocks[-1].edges()
        blocks=[i.to(self.model.device) for i in blocks]
        num_node=blocks[-1].num_dst_nodes()
        text_embedding, image_embedding = self.feature[idx,0].to(device),self.feature[idx,1].to(device)
        text_embedding=self.text_proj(text_embedding)
        image_embedding=self.img_proj(image_embedding)
        text_embedding, image_embedding = text_embedding[:num_node,].to(device), image_embedding[:num_node,].to(device)


        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        seq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(",",add_special_tokens=False)))
        seqq_embeds =self.word_embedding(torch.tensor(self.tokenizer.encode(";",add_special_tokens=False)))
        blank_embeds=self.word_embedding(torch.tensor(self.tokenizer.encode(" ",add_special_tokens=False)))
        #bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        questions = self.processor(self.prompt, add_special_tokens=False)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        batch_size = num_node
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        
        for i in range(batch_size):
            self.g_val=dgl.node_subgraph(self.g, self.train_idx.tolist()+[x["index"][i].item()])
            indices=torch.nonzero(self.g_val.ndata["index"]==x["index"][i].item())[0]
            one,two=self.generate_subgraphs(self.g_val, indices.item())
            o=[]
            t=[]
            for ix,(ij,labe) in enumerate(one):
                o.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                o.append(seq_embeds.to(self.model.device))
                o.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                o.append(seq_embeds.to(self.model.device))
                o.append(self.word_embedding(
                    torch.tensor(
                        self.processor(
                            self.map[labe.long().item()],add_special_tokens=False)["input_ids"]
                            )).to(self.model.device))
                o.append(seqq_embeds.to(self.model.device))
            for ix,(ij,labe) in enumerate(two):
                t.append(self.text_proj(self.feature[ij,[0]].to(self.model.device)))
                t.append(seq_embeds.to(self.model.device))
                t.append(self.img_proj(self.feature[ij,[1]].to(self.model.device)))
                t.append(seq_embeds.to(self.model.device))
                t.append(self.word_embedding(torch.tensor(self.processor(self.map[labe.long().item()],add_special_tokens=False)["input_ids"])).to(self.model.device))
                t.append(seqq_embeds.to(self.model.device))
            raw_text=self.processor(self.text[x["index"][i]], add_special_tokens=False)
            if self.max_txt_len==1:
                texte=self.word_embedding(torch.tensor(raw_text["input_ids"]).to(self.model.device)).mean(dim=0,keepdim=True)
            else:
                texte=self.word_embedding(torch.tensor(raw_text["input_ids"]).to(self.model.device))[:self.max_txt_len]
            if o==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    texte,
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    image_embedding[i].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4]).to(self.model.device))],
                    dim=0)
            elif t==[]:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    texte,
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    image_embedding[i].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4]).to(self.model.device))],
                    dim=0)
            else:
                inputs_embeds = torch.cat([
                    self.word_embedding(torch.tensor(questions["input_ids"][0]).to(self.model.device)),
                    texte,
                    #self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    #text_embedding[i][:self.max_txt_len],
                    self.word_embedding(torch.tensor(questions["input_ids"][1]).to(self.model.device)),
                    image_embedding[i].unsqueeze(0),
                    self.word_embedding(torch.tensor(questions["input_ids"][2]).to(self.model.device)),
                    torch.cat(o,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][3]).to(self.model.device)),
                    torch.cat(t,dim=0),
                    self.word_embedding(torch.tensor(questions["input_ids"][4]).to(self.model.device))],
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
                'label': [self.map[j.long().item()] for j in  blocks[-1].dstdata["label"]]}