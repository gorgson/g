import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import jsonlines
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import dgl
from transformers import CLIPProcessor, CLIPModel
import psutil
import os
from models import SAGE, GCN, MLP, MMGCN,GATv2,GAT
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizerFast,BertModel,ViTFeatureExtractor, ViTForImageClassification,ViTModel
class NodeClassificationDataset(object):
    def __init__(self, root: str,verbose: bool=True, device: str="cpu",bert_name: str = "bert-base-uncased",feat="clip",data_path="",save=False,trun=True):
        """
        Args:
            root (str): root directory to store the dataset folder.
            feat_name (str): the name of the node features, e.g., "t5vit".
            verbose (bool): whether to print the information.
            device (str): device to use.
        """
        clip_model_name = "/scratch/ys6310/clip-vit-base-patch16"
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model = CLIPModel.from_pretrained(clip_model_name).to("cuda")
        root = os.path.normpath(root)
        transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor()
        ])
        self.name = os.path.basename(root)
        self.verbose = verbose
        self.root = root
        df = pd.read_csv(f"{root}/{self.name}.csv")

        all_categories = sorted(df['second_category'].unique()) 

        gpth=f"{root}/{self.name}Graph.pt"
        graph = dgl.load_graphs(gpth)[0][0]

        self.label = graph.ndata['label']
        self.num_classes=max(self.label)+1
        self.device = device
        if self.verbose:
            print(f"Dataset name: {self.name}")

            print(f'Device: {self.device}')


        self.num_nodes = graph.num_nodes()
        batch_size=6000
        image=[]
        text=[]

        k=0
        self.label_to_second_category = list(dict(zip(df['label'], df['second_category'])).values())

        file_path = os.path.join(root,f"{self.name}Images")
        textfeat=torch.empty(0)
        imgfeat=torch.empty(0)
        att=torch.empty(0)
        if save:
            for stu in range(df.shape[0]):
            
                img_path = os.path.join(file_path, f"{stu}.jpg")
                #if not os.path.exists(os.path.join(data_path,self.name,"image_feat",f"{self.mapping[stu['asin']]}.pt")):
                        
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                    except Exception as e:
                        img = Image.new("RGB", (100, 100), "white") 
                else:
                    img = Image.new('RGB', (224, 224), (0, 0, 0))
                if img.mode == 'L':
                    img = Image.merge("RGB", (img, img, img))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                text.append(df.loc[stu,"caption"])
         
                image.append(img)

                if len(image)==batch_size:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            inputs=self.processor(text=text, images=image, return_tensors="pt", padding=True,truncation=True ).to("cuda")
                            outputs =self.model(**inputs)
                
                        if att.shape!=torch.empty(0).shape:
             
                            len_att = att.shape[1]
                            len_mask = inputs['attention_mask'].shape[1]
                            # 找到最大长度
                            max_len = max(len_att, len_mask)

                            # 如果长度不一致，对较短的 Tensor 补零
                            if len_att < max_len:
                                padding = torch.zeros(att.size(0), max_len - len_att)
                                att = torch.cat([att, padding], dim=1)
                                padding = torch.zeros(att.size(0), max_len - len_att,512)
                                textfeat= torch.cat([textfeat, padding], dim=1)
 

                            if len_mask < max_len:
                                padding = torch.zeros(inputs['attention_mask'].size(0), max_len - len_mask).cuda()
                                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], padding], dim=1)
                                padding = torch.zeros(inputs['attention_mask'].size(0), max_len - len_mask,512).cuda()
                                outputs.text_model_output.last_hidden_state= torch.cat([outputs.text_model_output.last_hidden_state, padding], dim=1)
                        print(att.shape)
                        print(inputs['attention_mask'].shape)
                        att = torch.cat([att, inputs['attention_mask'].cpu()], dim=0)
                        imgfeat=torch.cat([imgfeat,self.model.visual_projection(outputs.vision_model_output.last_hidden_state).cpu()],dim=0)
                        textfeat=torch.cat([textfeat,self.model.text_projection(outputs.text_model_output.last_hidden_state).cpu()],dim=0)
                        print(imgfeat.shape)
                        print(textfeat.shape)
                        image=[]
                        text=[]
        
                    k+=batch_size
                    print(k)

            if image!=[]:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        inputs=self.processor(text=text, images=image, return_tensors="pt", padding=True,truncation=True ).to("cuda")
                        outputs =self.model(**inputs)
            
                    if att.shape!=torch.empty(0).shape:
            
                        len_att = att.shape[1]
                        len_mask = inputs['attention_mask'].shape[1]
                        # 找到最大长度
                        max_len = max(len_att, len_mask)

                        # 如果长度不一致，对较短的 Tensor 补零
                        if len_att < max_len:
                            padding = torch.zeros(att.size(0), max_len - len_att)
                            att = torch.cat([att, padding], dim=1)
                            padding = torch.zeros(att.size(0), max_len - len_att,512)
                            textfeat= torch.cat([textfeat, padding], dim=1)


                        if len_mask < max_len:
                            padding = torch.zeros(inputs['attention_mask'].size(0), max_len - len_mask).cuda()
                            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], padding], dim=1)
                            padding = torch.zeros(inputs['attention_mask'].size(0), max_len - len_mask,512).cuda()
                            outputs.text_model_output.last_hidden_state= torch.cat([outputs.text_model_output.last_hidden_state, padding], dim=1)
                    print(att.shape)
                    print(inputs['attention_mask'].shape)
                    att = torch.cat([att, inputs['attention_mask'].cpu()], dim=0)
                    imgfeat=torch.cat([imgfeat,self.model.visual_projection(outputs.vision_model_output.last_hidden_state).cpu()],dim=0)
                    textfeat=torch.cat([textfeat,self.model.text_projection(outputs.text_model_output.last_hidden_state).cpu()],dim=0)
                    print(imgfeat.shape)
                    print(textfeat.shape)
                    image=[]
                    text=[]
    
                k+=batch_size
                print(k)
                    
            torch.save(imgfeat,os.path.join(root,"cimg_feat.pt"))
            print("img")
            print(imgfeat.dtype)

            torch.save(textfeat,os.path.join(root,"ctext_feat.pt"))
            print("text")
            print(textfeat.dtype)

            torch.save(att,os.path.join(root,"catt.pt"))
            print("att")
            print(att.dtype)

            return 
  
        if not trun:
            node_ids=torch.load(os.path.join(root,"text_feat.pt"))
        else:
            node_ids=torch.load(os.path.join(root,"ctext_feat.pt"))

        src, dst =graph.edges()
        self.graph = dgl.graph((src, dst), num_nodes=self.num_nodes).to(self.device)
        self.graph.ndata['image_feat'] = torch.load(os.path.join(root,"cimg_feat.pt")).to(self.device).contiguous()
        self.graph.ndata["text_feat"]=node_ids.to(self.device).contiguous()


        self.graph.ndata['attention_mask']=torch.load(os.path.join(root,"catt.pt")).to(self.device)
        self.text=list(df["text"])
        node_split_path = os.path.join(root, 'split.pt')
        self.node_split = self.split_graph(self.num_nodes,0.6,0.2)
        
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)

        train_mask[self.node_split[0]] = True
        val_mask[self.node_split[1]] = True
        test_mask[self.node_split[2]] = True
 
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        self.graph.ndata["index"]=torch.arange(self.num_nodes)
        self.graph.ndata["label"]=self.label

        if self.name=="Movies":
            target_domain='Movies & TV'
            co="co-viewed"
        elif self.name=="Toys":
            target_domain='Toys & Games'
            co="co-buyed"
        elif self.name=="RedditS":
            target_domain='Reddit'
        elif self.name=="Grocery":
            target_domain='Grocery & Gourmet Food' 


        self.prompt=[
            f"I’m starting a node classification task in the Amazon-{target_domain} dataset. Each node represents a {target_domain} with text and image features, and edges indicate {co} relation. Given a target node, the text features are ",
            #<text sequence> 
            " and the image features are ",
            #<img sequence> (如果哪一个没提供就把对应部分描述删掉)
            ". The neighbors are described in the following template: <text feature>, <image feature>, and <label>. It has the following neighbors at hop 1: ",
            #N1:<1-hop neighbor 1 text feature>, <1-hop neighbor 1 image feature>, <1-hop neighbor 1 label>
            #N2:<1-hop neighbor 2 text feature>, <1-hop neighbor 2 image feature>, <1-hop neighbor 2 label>
            #N3: ...........
            " It has the following neighbors at hop 2: ",
            #N1:<2-hop neighbor 1 text feature>, <2-hop neighbor 1 image feature>, <2-hop neighbor 1 label>
            #N2:<2-hop neighbor 2 text feature>, <2-hop neighbor 2 image feature>, <2-hop neighbor 2 label>
            #N3: ...........
            f" Based on the information provided, please classify the target movie into one of the following categories: {all_categories}. "
            ]

        '''
         After step-by-step deep thinking, we can explain why it is classified as this label from three aspects: text features, image features, and neighbor information. The reasons are as follows:
        self.prompt = [
 f"In the movie graph, each node represents a movie, and each movie has two modalities of features: text and image feature.An edge between two nodes indicates that the corresponding movies have been viewed together by customers. Each movie can be classified into one of the following categories: {all_categories}. Given a target movie, the text features are ",
 #<text sequence> 
" and the image features are ",
#<img sequence>
" The 1-hop neighbors of the movie (movies directly related by shared views) include(the format for each neighbor is: text features, image features, and the category to which the movie belongs): " , 
#<1-hop neighbor 1 text feature>, <1-hop neighbor 1 image feature>, <1-hop neighbor 1 label>
#<1-hop neighbor 2 text feature>, <1-hop neighbor 2 image feature>, <1-hop neighbor 2 label>
". The 2-hop neighbors of the movie (movies indirectly related by shared neighbors) include(the format for each neighbor is: text features, image features, and the category to which the movie belongs): ",
#"<2-hop neighbor 1 text feature>, <2-hop neighbor 1 image feature>, <2-hop neighbor 1 label>, ",
#"<2-hop neighbor 2 text feature>, <2-hop neighbor 2 image feature>, <2-hop neighbor 2 label>, ",    
 ". Please classify the target movie into one of the categories provided above and tell me its category label."
 ]
 '''
    def get_idx_split(self):
        return self.node_split

    def split_graph(self, nodes_num, train_ratio, val_ratio):
        np.random.seed(42)  # 设置随机种子，确保每次拆分一致
        indices = np.random.permutation(nodes_num)  # 随机打乱节点索引

        # 计算训练集、验证集和测试集的大小
        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)

        # 获取各个数据集的节点索引
        train_ids = torch.tensor(indices[:train_size], dtype=torch.long)
        val_ids = torch.tensor(indices[train_size:train_size + val_size], dtype=torch.long)
        test_ids = torch.tensor(indices[train_size + val_size:], dtype=torch.long)

        # 返回大小和对应的 Torch Tensor 类型的节点索引
        return train_ids, val_ids, test_ids


    def __getitem__(self, idx: int):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph
    
    def __len__(self):
        return 1
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

# borrowed from OGB
class NodeClassificationEvaluator:
    def __init__(self, eval_metric: str):
        """
        Args:
            eval_metric (str): evaluation metric, can be "rocauc" or "acc".
        """
        self.num_tasks = 1
        self.eval_metric = eval_metric


    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'acc':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks should be {} but {} given'.format(self.num_tasks, y_true.shape[1]))

            return y_true, y_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))


    def eval(self, input_dict):

        if self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator\n'
        if self.eval_metric == 'rocauc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores score values (for computing ROC-AUC),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        elif self.eval_metric == 'acc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores predicted class label (integer),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator\n'
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'acc':
            desc += '{\'acc\': acc}\n'
            desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(self.num_tasks)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list)/len(rocauc_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}
