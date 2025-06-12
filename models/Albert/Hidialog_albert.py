import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from .modeling_albert import AlbertPreTrainedModel, AlbertModel


class GTN(nn.Module):
    '''Graph Transformer Networks - adapted from https://github.com/seongjunyun/Graph_Transformer_Networks'''
    
    def __init__(self, etypes, num_channels, w_in, w_out, num_layers, norm=True, num_class=None):
        super(GTN, self).__init__()
        self.etypes = etypes
        self.num_edge = len(etypes) + 1
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(self.num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(self.num_edge, num_channels, first=False))
        
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
    
        if num_class is not None:
            self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)
            self.linear2 = nn.Linear(self.w_out, self.num_class)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self, X, H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(), X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i == 0:
                H_ = self.norm(H[i, :, :]).unsqueeze(0)
            else:
                H_ = torch.cat((H_, self.norm(H[i, :, :]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor)).to(H.device)
        else:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor)).to(H.device) + torch.eye(H.shape[0]).type(torch.FloatTensor).to(H.device)
        
        deg = torch.sum(H, dim=1).to(H.device)
        deg_inv = (deg + 1e-8).pow(-1)
        deg_inv[deg == 0] = 0
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv * torch.eye(H.shape[0]).type(torch.FloatTensor).to(H.device)
        H = torch.mm(deg_inv, H)
        H = H.t()
        return H
    
    def adj(self, graph):
        num_nodes = graph.number_of_nodes('node')
        for k, etype in enumerate(self.etypes):
            A_k = graph.adj(etype=etype)
            if k == 0:
                A = A_k.to_dense().float().unsqueeze(-1)
            else:
                A = torch.cat([A, A_k.to_dense().float().unsqueeze(-1)], dim=-1)

        A = torch.cat([A, torch.eye(num_nodes).float().to(A.device).unsqueeze(-1)], dim=-1)
        return A

    def forward(self, graph, X, target_x=None, target=None, output_gtn_adjacency=False):
        A = self.adj(graph)
        A = A.unsqueeze(0).permute(0, 3, 1, 2) 
        Ws = []

        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        
        for i in range(self.num_channels):
            if i == 0:
                X_ = F.relu(self.gcn_conv(X, H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X, H[i]))
                X_ = torch.cat((X_, X_tmp), dim=1)
        
        if target is None:
            Hs = torch.stack([self.norm(H_, add=True) for H_ in H]).detach().cpu() if output_gtn_adjacency else None
            return X_, Ws, Hs
        else:
            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_[target_x])
            loss = self.loss(y, target)
            return loss, y, Ws


class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a, b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach().cpu(),
                 (F.softmax(self.conv2.weight, dim=1)).detach().cpu()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_, a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach().cpu()]
        
        return H, W


class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = A.to(self.weight.device)
        A = torch.sum(A * F.softmax(self.weight, dim=1), dim=1)
        return A


class HiDialog_Albert(AlbertPreTrainedModel):
    """
    HiDialog model with ALBERT backbone for dialog relation extraction.
    Combines ALBERT with Graph Transformer Networks (GTN) to model dialog structure.
    """
    
    def __init__(self, 
                 config, 
                 num_labels, 
                 gtn_out=1024, 
                 activation='relu', 
                 gtn_num_channels=3,
                 gtn_num_layers=2,
                 gcn_dropout=0.6,
                 dataset='DialogRE',
                 intra_turn_only=False):
        
        super().__init__(config)
        
        self.num_labels = num_labels
        self.albert = AlbertModel(config)
        self.dataset = dataset
        self.intra_turn_only = intra_turn_only
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.intra_turn_only:
            # Simple classification without GTN
            self.classifier = nn.Linear(gtn_out, self.num_labels)
        else:
            # GTN for modeling dialog structure
            self.etypes = ['dialog', 'sequence', 'entity', 'speaker']
            self.gtn = GTN(self.etypes, gtn_num_channels, config.hidden_size, gtn_out, gtn_num_layers)
            self.classifier = nn.Linear(gtn_out * 3 * gtn_num_channels, self.num_labels)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        speaker_ids=None, 
        graphs=None, 
        hidialog_mask=None,
        cls_indices=None,
        tran_ids=None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get ALBERT outputs
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask, #hidialog_mask if hidialog_mask is not None else attention_mask, 
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_outputs = outputs[0]  # [batch_size, seq_len, hidden_size]
        pooled_outputs = outputs[1] if len(outputs) > 1 else None
        
        if not self.intra_turn_only and graphs is not None:
            # Use GTN for dialog structure modeling
            graph_output = []
            
            for i, graph in enumerate(graphs):
                # Extract CLS features for each turn
                sequence_output = sequence_outputs[i]
                mask = cls_indices[i].unsqueeze(-1).expand(-1, sequence_output.shape[-1])
                features_graph = torch.masked_select(sequence_output, mask).view(-1, sequence_output.shape[-1])
                
                # Apply GTN
                features_graph, Ws, Hs = self.gtn(graph, features_graph, output_gtn_adjacency=output_attentions)

                # Extract dialog, object, and subject node features
                dialog_node = features_graph[0]
                if tran_ids is None:
                    obj_node, sbj_node = features_graph[-2], features_graph[-1]
                else:
                    obj_node, sbj_node = features_graph[-1], features_graph[int(tran_ids[i])]
                
                # Integrate features
                integrated_output = torch.cat((dialog_node, obj_node, sbj_node), dim=-1)
                graph_output.append(integrated_output)

            graph_output = torch.stack(graph_output)
            pooled_output = self.dropout(graph_output)
        
        else:
            # Simple intra-turn classification (use CLS token)
            pooled_output = sequence_outputs[:, 0]  # [CLS] token
            pooled_output = self.dropout(pooled_output)
            
        # Final classification
        logits = self.classifier(pooled_output)
        logits = logits.view(-1, self.num_labels)

        # Compute loss based on dataset type
        if labels is not None:
            if self.dataset not in ['MRDA', 'MuTual']:
                # Multi-label classification (e.g., DialogRE)
                loss_fct = BCEWithLogitsLoss()
                labels = labels.view(-1, self.num_labels)
                loss = loss_fct(logits, labels)
                return loss, logits
            elif self.dataset == 'MRDA':
                # Single-label classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, logits
            elif self.dataset == 'MuTual':
                # Multiple choice classification
                logits = logits.view(-1, 4)  # num choices
                loss_fct = CrossEntropyLoss()
                labels = torch.argmax(labels.view(-1, 4), dim=1)  # locate answer ids
                loss = loss_fct(logits, labels)
                logits = logits.view(-1, self.num_labels)
                return loss, logits         
        else:
            return logits, None