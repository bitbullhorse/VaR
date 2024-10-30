from torch import Tensor
from torch.nn import LayerNorm, init
from torch.nn.modules import transformer
import string
from typing import List, Optional
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pandas.core.frame
import math
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler, StandardScaler

torch.autograd.set_detect_anomaly(True)

device = 'cuda'

INDEX = ['收盘价_Clpr', '开盘价_Oppr', '最高价_Hipr', '最低价_Lopr', '复权价1(元)_AdjClpr1', '复权价2(元)_AdjClpr2', '成交量_Trdvol',\
               '成交金额_Trdsum','日振幅(%)_Dampltd','总股数日换手率(%)_DFulTurnR', '流通股日换手率(%)_DTrdTurnR', '日收益率_Dret', '日资本收益率_Daret', \
               '等权平均市场日收益率_Dreteq','流通市值加权平均市场日收益率_Drettmv', '总市值加权平均市场日收益率_Dretmc', '等权平均市场日资本收益率_Dareteq',\
               '总市值加权平均日资本收益_Daretmc', '日无风险收益率_DRfRet', '市盈率_PE']

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2)) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # 添加到模块的缓冲区中，不会被视为模型参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码添加到序列嵌入表示上
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CustomDatasetALL(Dataset):
    def __init__(self, dataframe: pandas.core.frame.DataFrame, tics: list, max_size: int = 30):
        self.max_size = max_size
        tmp_df = dataframe.copy()
        seq_df = tmp_df[tmp_df['tic'] == tics[0]]
        seq_df = seq_df.drop(columns=['tic', 'day', 'turbulence'])
        flag = 1
        for tic in tics:
            if tic == tics[0]:
                continue
            tic_df = tmp_df[tmp_df['tic'] == tic]
            tic_df = tic_df.drop(columns=['tic', 'day', 'turbulence'])
            name_list = ['date']
            for i in range(100):
                name_list.append(f"_{flag} __{i}_")
            tic_df.columns = name_list[0:len(tic_df.columns)]
            if len(tic_df) % max_size != 0:
                n = len(tic_df) % max_size
                tic_df = tic_df[:-n]
            seq_df = pd.merge(seq_df, tic_df, on='date')
            flag += 1
        self.seq_df = seq_df.drop(columns='date')

    def __getitem__(self, index):
        src_df = self.seq_df[index * self.max_size: (index + 1) * self.max_size]
        tgt_df = self.seq_df[index * self.max_size + 1: (index + 1) * self.max_size + 1]
        # tgt_df = self.seq_df[(index + 1) * self.max_size: (index + 2) * self.max_size]
        return torch.tensor(src_df.values, dtype=torch.float64), torch.tensor(tgt_df.values, dtype=torch.float64)

    def __len__(self):
        return int(len(self.seq_df) / self.max_size) - 1


class CostomDatasetSingle(Dataset):
    def __init__(self, dataframe: pandas.core.frame.DataFrame, tic: str, max_size: int = 12,dmodel : int = 14, dtype=torch.float64, device='cuda'):

        self.max_size = max_size
        tmp_df = dataframe.copy()
        scaler = StandardScaler()
        tmp_df[['open', 'high', 'low', 'close', 'volume', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix']] = \
            scaler.fit_transform(tmp_df[['open', 'high', 'low', 'close', 'volume', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix']])
        seq_df = tmp_df[tmp_df['tic'] == tic]
        if(dmodel == 14):
            seq_df = seq_df.drop(columns=['tic', 'day', 'turbulence'])
        else:
            seq_df = seq_df.drop(columns=['tic'])
        self.close_df = seq_df['close']
        self.seq_df = seq_df.drop(columns='date')

    def __getitem__(self, index):
        src_df = self.seq_df[index: index +  self.max_size]
        tgt_df = self.seq_df[(index + 1) * self.max_size: (index + 2) * self.max_size]
        result = torch.zeros(2, dtype=torch.float64,device=device)
        if self.close_df.iloc[index + self.max_size] - self.close_df.iloc[index + self.max_size - 1] > 0:
            result[0] = 1
        else:
            result[1] = 1
        return torch.tensor(src_df.values, dtype=torch.float64, device=device), result
        # result = 0
        # if self.close_df.iloc[(index + 1) * self.max_size] - self.close_df.iloc[(index + 1) * self.max_size - 1] > 0:
        #     result = 1
        # return torch.tensor(src_df.values, dtype=torch.float64, device=device), torch.tensor([result], dtype=torch.float64, device=device)


    def __len__(self):
        return int(len(self.seq_df) - self.max_size)

class CustomDataset(Dataset):
    def __init__(self, dataframe: pandas.core.frame.DataFrame, max_size: int = 30):
        self.max_size = max_size
        tmp_df = dataframe.copy()
        tmp_df = tmp_df.dropna()
        minmaxscaler = MinMaxScaler()
        tmp_df[INDEX] = minmaxscaler.fit_transform(tmp_df[INDEX])
        self.seq_df = tmp_df
        self.close_df = tmp_df['收盘价_Clpr']
        # self.close_df = self.close_df.dropna()
        # self.seq_df = self.seq_df.dropna()

    def __getitem__(self, index):
        src_df = self.seq_df[index: index + self.max_size]
        tgt_df = self.seq_df[index + 1: index + self.max_size + 1]
        Cl_pr = self.close_df[index + self.max_size]
        return torch.tensor(src_df.values, dtype=torch.float64, device=device), torch.tensor(tgt_df.values,
                                                                                             dtype=torch.float64,
                                                                                             device=device), torch.unsqueeze(torch.tensor(Cl_pr, dtype=torch.float64, device=device), dim=0)
    def __len__(self):
        return int(min(len(self.seq_df  - self.max_size), len(self.close_df) - self.max_size))
        # return int(len(self.seq_df  - self.max_size))


class Custom_iTransformer_Dataset(Dataset):
    def __init__(self, dataframe: pandas.core.frame.DataFrame, seq_len: int=30, pred_len: int=2):
        self.seq_len = seq_len
        self.pred_len = pred_len
        tmp_df = dataframe.copy()
        minmaxscaler = MinMaxScaler()
        # tmp_df[INDEX] = minmaxscaler.fit_transform(tmp_df[INDEX])
        tmp_df = tmp_df.dropna()
        self.seq_df = tmp_df
        self.close_df = tmp_df['收盘价_Clpr']

    def __getitem__(self, index):
        seq = self.seq_df[index: index + self.seq_len]
        label = self.seq_df[index + self.seq_len: index + self.seq_len + self.pred_len]
        return torch.tensor(seq.values, dtype=torch.float64, device=device), torch.tensor(label.values, dtype=torch.float64, device=device),

    def __len__(self):
        return int(len(self.seq_df) - self.seq_len - self.pred_len + 1)



class CustomDatasetSingleRegression(Dataset):
    def __init__(self, dataframe: pandas.core.frame.DataFrame, tic: str, max_size: int = 12,dmodel : int = 14, dtype=torch.float64, device='cuda'):
        self.max_size = max_size
        stdscaler = StandardScaler()
        minmaxscaler = MinMaxScaler()
        tmp_df = dataframe.copy()
        tmp_df[['open', 'high', 'low', 'close', 'volume', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                'close_30_sma', 'close_60_sma', 'vix']] = \
            minmaxscaler.fit_transform(tmp_df[
                                     ['open', 'high', 'low', 'close', 'volume', 'macd', 'boll_ub', 'boll_lb', 'rsi_30',
                                      'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix']])
        seq_df = tmp_df[tmp_df['tic'] == tic]
        if(dmodel == 14):
            seq_df = seq_df.drop(columns=['tic', 'day', 'turbulence'])
        else:
            seq_df = seq_df.drop(columns=['tic'])
        self.close_df = seq_df['close']
        self.seq_df = seq_df.drop(columns='date')

    def __getitem__(self, index):
        src_df = self.seq_df[index: index +  self.max_size]
        tgt_df = self.seq_df.iloc[index + self.max_size]

        return torch.tensor(src_df.values, dtype=torch.float64, device=device), torch.tensor(tgt_df.values, dtype=torch.float64, device=device)
        # result = 0
        # if self.close_df.iloc[(index + 1) * self.max_size] - self.close_df.iloc[(index + 1) * self.max_size - 1] > 0:
        #     result = 1
        # return torch.tensor(src_df.values, dtype=torch.float64, device=device), torch.tensor([result], dtype=torch.float64, device=device)


    def __len__(self):
        return int(len(self.seq_df) - self.max_size)


class CustomDatasetSingleTransRegression(Dataset):
    def __init__(self, dataframe: pandas.core.frame.DataFrame, tic: str, max_size: int = 12,dmodel : int = 14, dtype=torch.float64, device='cuda'):
        self.max_size = max_size
        stdscaler = StandardScaler()
        minmaxscaler = MinMaxScaler()
        tmp_df = dataframe.copy()
        tmp_df[['open', 'high', 'low', 'close', 'volume', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                'close_30_sma', 'close_60_sma', 'vix']] = \
            minmaxscaler.fit_transform(tmp_df[
                                     ['open', 'high', 'low', 'close', 'volume', 'macd', 'boll_ub', 'boll_lb', 'rsi_30',
                                      'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix']])
        seq_df = tmp_df[tmp_df['tic'] == tic]
        if(dmodel == 14):
            seq_df = seq_df.drop(columns=['tic', 'day', 'turbulence'])
        else:
            seq_df = seq_df.drop(columns=['tic'])
        self.close_df = seq_df['close']
        self.seq_df = seq_df.drop(columns='date')

    def __getitem__(self, index):
        src_df = self.seq_df[index: index +  self.max_size]
        tgt_df = self.seq_df[index + 1: index +  self.max_size + 1]

        return torch.tensor(src_df.values, dtype=torch.float64, device=device), torch.tensor(tgt_df.values, dtype=torch.float64, device=device)
        # result = 0
        # if self.close_df.iloc[(index + 1) * self.max_size] - self.close_df.iloc[(index + 1) * self.max_size - 1] > 0:
        #     result = 1
        # return torch.tensor(src_df.values, dtype=torch.float64, device=device), torch.tensor([result], dtype=torch.float64, device=device)


    def __len__(self):
        return int(len(self.seq_df) - self.max_size)


class TransformerDecoderLayer1(transformer.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
    ) -> Tensor:

        x = memory
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), x, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, x, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))
        return x


# 自定义Transformer模型解码器
class Decoder_Only_Transformer1(nn.Module):
    def __init__(self, d_model, nhead, batch_first=True, dtype=torch.float64, seq_len=12, nlayers=6,
                 layer_norm_eps: float = 1e-5,bias: bool = True):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Decoder_Only_Transformer1, self).__init__()
        self.position_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        decoder_layer = TransformerDecoderLayer1(d_model=d_model, nhead=nhead, batch_first=batch_first, dtype=dtype,
                                                 norm_first=False)
        layernorm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers, norm=layernorm)
        self.linear = nn.Linear(d_model, d_model, dtype=dtype)

    def forward(self, mem, tgt):
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(mem.shape[1], dtype=torch.float64)
        tgt_mask = tgt_mask.to(device)
        mem = self.position_encoder(mem)
        output = self.linear(self.decoder(tgt=tgt, memory=mem, tgt_mask=None))
        return output


class Transformer_model(nn.Module):
    def __init__(self, d_model, nhead, batch_first=True, dtype=torch.float64, seq_len=12, nlayers=6):
        super(Transformer_model, self).__init__()
        self.position_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, batch_first=batch_first, dtype=dtype,
                                          num_encoder_layers=nlayers, num_decoder_layers=nlayers)
        self.linear = nn.Linear(d_model, 2 * d_model, dtype=dtype, )
        self.linear2 = nn.Linear(2 * d_model, d_model, dtype=dtype, )
    def forward(self, src, tgt, has_mask=True):
        if has_mask:
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(src.shape[1], dtype=torch.float64)
            tgt_mask = tgt_mask.to(device)
        else:
            tgt_mask = None
        src = self.position_encoder(src)
        tgt = self.position_encoder(tgt)
        output = self.linear(self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask))
        return output


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init.normal_(m.weight, mean=0, std=0.01)
    elif isinstance(m, nn.LayerNorm):
        init.constant_(m.bias, 0)
        init.constant_(m.weight, 1.0)

class Transformer_model_Cp(nn.Module):
    def __init__(self, d_model, nhead, batch_first=True, dtype=torch.float64, seq_len=12, nlayers=6):
        super(Transformer_model_Cp, self).__init__()
        self.position_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, batch_first=batch_first, dtype=dtype,
                                          num_encoder_layers=nlayers, num_decoder_layers=nlayers)
        self.transformer.apply(init_weights)
        self.linear = nn.Linear(d_model, 512, dtype=dtype, )
        self.linear2 = nn.Linear(512, 1, dtype=dtype, )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(512).double()  # 添加 Layer Normalization

    def forward(self, src, tgt, has_mask=True):
        if has_mask:
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(src.shape[1], dtype=torch.float64)
            tgt_mask = tgt_mask.to(device)
        else:
            tgt_mask = None
        src = self.position_encoder(src)
        tgt = self.position_encoder(tgt)
        output = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)[:,-1,:]
        output = self.linear(output)
        output = self.layer_norm(output)
        output = self.relu(output)
        output = self.linear2(output)
        return output

# class Transformer_model(nn.Module):
#     def __init__(self, d_model, nhead, batch_first=True, dtype=torch.float64, seq_len=12, nlayers=6):
#         super(Transformer_model, self).__init__()
#         self.position_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
#         self.linear1 = nn.Linear(d_model, 512,dtype=dtype)
#         self.transformer = nn.Transformer(d_model=512, nhead=nhead, batch_first=batch_first, dtype=dtype,
#                                           num_encoder_layers=nlayers, num_decoder_layers=nlayers)
#         self.linear = nn.Linear(512, d_model, dtype=dtype, )
#
#     def forward(self, src, tgt, has_mask=True):
#         if has_mask:
#             tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(src.shape[1], dtype=torch.float64)
#             tgt_mask = tgt_mask.to(device)
#         else:
#             tgt_mask = None
#         src = self.position_encoder(src)
#         tgt = self.position_encoder(tgt)
#         src = F.elu(self.linear1(src))
#         tgt = F.elu(self.linear1(tgt))
#         src = F.layer_norm(src, [512])
#         tgt = F.layer_norm(tgt, [512])
#         output = F.relu(self.linear(self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)))
#         return output


class TransformerDeLSTMPredictor(nn.Module):
    def __init__(self, d_model, nhead, batch_first=True, dtype=torch.float64, seq_len=12, nlayers=6,layer_norm_eps: float = 1e-5,bias: bool = True):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDeLSTMPredictor, self).__init__()
        self.position_encoder = PositionalEncoding(d_model, dropout=0, max_len=seq_len)
        decoder_layer = TransformerDecoderLayer1(d_model=d_model, nhead=nhead, batch_first=batch_first, dtype=dtype,
                                                 norm_first=False)
        layernorm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias,**factory_kwargs)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers, norm=layernorm)
        self.Linear = nn.Linear(d_model, 128, dtype=dtype)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=batch_first,num_layers=8,**factory_kwargs)
        self.predictor = nn.Linear(128, 2, dtype=dtype)

    def forward(self, tgt, src):
        x = self.position_encoder(src)
        x = self.decoder(tgt=tgt, memory=x)
        x = self.Linear(x)
        x, hidden = self.lstm(x)
        x = x[:, -1, :]
        out = self.predictor(x)
        return out

class TransformerDeLSTMRegression(nn.Module):
    def __init__(self, d_model, nhead, batch_first=True, dtype=torch.float64, seq_len=12, nlayers=6,layer_norm_eps: float = 1e-5,bias: bool = True):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDeLSTMRegression, self).__init__()
        self.position_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        decoder_layer = TransformerDecoderLayer1(d_model=d_model, nhead=nhead, batch_first=batch_first, dtype=dtype,
                                                 norm_first=False)
        layernorm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias,**factory_kwargs)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers, norm=layernorm)
        self.Linear = nn.Linear(d_model, 128, dtype=dtype)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=batch_first, num_layers=8,**factory_kwargs)
        self.predictor = nn.Linear(128, d_model, dtype=dtype)

    def forward(self, tgt, src):
        x = self.position_encoder(src)
        # x = src
        x = self.decoder(tgt=x, memory=x)
        x = self.Linear(x)
        x, hidden = self.lstm(x)
        x = x[:,-1, :]
        out = self.predictor(x)
        return out


class PureLSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, dtype=torch.float64, num_layers=8, dropout=0):
        super(PureLSTMRegression, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first, num_layers=num_layers,dtype=dtype, dropout=dropout)
        self.Linear = nn.Linear(input_size, input_size, bias=False, dtype=dtype)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        h0, c0 = h0.to(device), c0.to(device)
        x, hidden = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.Linear(x)
        return x


class RNNRegression(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, dtype=torch.float64, num_layers=6, dropout=0):
        super(RNNRegression, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first, num_layers=num_layers, dtype=dtype, dropout=dropout)
        self.Linear = nn.Linear(hidden_size, input_size, bias=False, dtype=dtype)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        x, hidden = self.rnn(x, h0)
        x = x[:, -1, :]
        x = self.Linear(x)
        return x


# class TransformerGAN(nn.Module):
