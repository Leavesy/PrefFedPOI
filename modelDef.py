import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(torch.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = torch.sin(dlat/2.0)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2.0)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    return c * 6367 * 1000

def haversine_bat(s_s):
    res=[]
    Len=s_s.shape[1]
    for ind in range(s_s.shape[0]):
        r1=torch.repeat_interleave(s_s[ind],Len,dim=0)
        r2=s_s[ind].repeat(Len,1)
        if len(res)==0:
            res=haversine(r1[:,0], r1[:,1], r2[:,0], r2[:,1]).reshape(-1,Len).unsqueeze(0)
        else:
            res=torch.cat([res,haversine(r1[:,0], r1[:,1], r2[:,0], r2[:,1]).reshape(-1,Len).unsqueeze(0)],dim=0)
    return res

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.max_len=max_len
        self.d_model=d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

    def forward(self, t_s, device):
        self.position = torch.arange(0, t_s.shape[1]).to(device)
        t_diff=torch.diff(t_s).masked_fill(torch.diff(t_s)<0,0)
        t_inter=t_diff/(t_diff.sum(dim=1).unsqueeze(1))
        t_inter=torch.cat([torch.zeros(t_s.shape[0],1).to(device),t_inter],dim=1)
        t_inter_p=(self.position+t_inter.cumsum(dim=1)).unsqueeze(-1)
        t_inter_p_dt=t_inter_p * (self.div_term.to(device))
        pe = torch.zeros(t_s.size(0), t_s.shape[1], self.d_model)
        pe.to(device)
        pe[:,:,0::2] = torch.sin(t_inter_p_dt)
        pe[:,:,1::2] = torch.cos(t_inter_p_dt)

        return Variable(pe,requires_grad=False).to(device)


class SpatialTemporalMask(nn.Module):
    def __init__(self, maxinter_t, maxinter_d, max_len):
        super(SpatialTemporalMask, self).__init__()
        self.maxinter_t=maxinter_t
        self.maxinter_d=maxinter_d
        self.max_len=max_len
        self.SoftMax=nn.Softmax(dim=-1)

    def forward(self, t_s, s_s, val_len, device):
        for ind in range(len(val_len)):
            if ind==0:
                padding_mat=F.pad(torch.tril(torch.ones(val_len[ind],val_len[ind]),diagonal=0),[0,t_s.shape[1]-val_len[ind],0,t_s.shape[1]-val_len[ind]],'constant',0).unsqueeze(0).to(device)
            else:
                padding_mat=torch.cat([padding_mat,F.pad(torch.tril(torch.ones(val_len[ind],val_len[ind]),diagonal=0),[0,t_s.shape[1]-val_len[ind],0,t_s.shape[1]-val_len[ind]],'constant',0).unsqueeze(0).to(device)],dim=0)
        t_matr=t_s.unsqueeze(-1)-t_s.unsqueeze(-2)
        t_matr=torch.abs(t_matr*padding_mat)
        t_matr=(t_matr<=self.maxinter_t)*t_matr+(t_matr>self.maxinter_t)*self.maxinter_t
        s_matr=haversine_bat(s_s)
        s_matr=torch.abs(s_matr*padding_mat)
        s_matr=(s_matr<=self.maxinter_d)*s_matr+(s_matr>self.maxinter_d)*self.maxinter_d
        r_matr=F.normalize(t_matr,dim=-1)+F.normalize(s_matr,dim=-1)
        r_matr=(r_matr.reshape(r_matr.shape[0],-1).max(dim=1).values).unsqueeze(-1).unsqueeze(-1)-r_matr
        r_matr=self.SoftMax(r_matr*padding_mat)
        r_matr=r_matr.masked_fill(~padding_mat.bool(),-1e9)

        return Variable(r_matr,requires_grad=False).to(device)

class TimSloPref(nn.Module):
    def __init__(self, num_time_slots, d_model, keep_rate):
        super(TimSloPref, self).__init__()
        self.preference=Variable(torch.zeros(num_time_slots+1,d_model),requires_grad=False)
        self.keep_rate=keep_rate

    def forward(self, time_slots, upd_time_slot=-1, preference=-1):
        if upd_time_slot!=-1:
            self.preference[upd_time_slot]=self.preference[upd_time_slot]*self.keep_rate+preference*(1-self.keep_rate)
        for ind in range(time_slots.shape[0]):
            if ind==0:
                slots_pref=self.preference[time_slots[ind]].unsqueeze(0)
            else:
                slots_pref=torch.cat([slots_pref,self.preference[time_slots[ind]].unsqueeze(0)],dim=0)

        return Variable(slots_pref,requires_grad=False)


class Attention(nn.Module):

    def __init__(self, input_Dim):
        super(Attention, self).__init__()
        self.mask = None
        self.output_Dim = 64
        self.default_input_dim = input_Dim

        self.attentionLayer = nn.Sequential(
            nn.Linear(self.default_input_dim, self.output_Dim),
            nn.ReLU(),
            nn.Linear(self.output_Dim, 1))

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, input):
        batch_size = input.size(0)
        hidden_size = input.size(2)
        input_size = input.size(1)
        attn = self.attentionLayer(input)

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        output = torch.bmm(attn, input)
        return torch.squeeze(output, 1), torch.squeeze(attn, 1)


class Parameter(object):
    def __init__(self,LocSize,CatSize,TimSloNum,EmbDim):
        self.LocSize=LocSize
        self.CatSize=CatSize
        self.TimSloNum=TimSloNum
        self.EmbDim=EmbDim


class WeigBCELoss(nn.Module):
    def __init__(self, temp):
        super(WeigBCELoss, self).__init__()
        self.temp = temp

    def forward(self, pos_scores, neg_scores):
        pos_part = F.logsigmoid(pos_scores)
        weig = F.softmax(neg_scores / self.temp, dim=-1)
        neg_part = (weig*torch.log((1-torch.sigmoid(neg_scores))+1e-9)).sum(-1)
        loss = (-(pos_part + neg_part).sum(-1))/pos_scores.shape[0]

        return loss


class LocalModel(nn.Module):
    def __init__(self, para):
        super(LocalModel, self).__init__()
        self.para=para
        self.POIEmb=nn.Embedding(para.LocSize, para.EmbDim)
        self.CatEmb=nn.Embedding(para.CatSize, para.EmbDim)
        self.PosEnc=PositionalEncoding(para.EmbDim,para.MaxLen)
        self.STMask=SpatialTemporalMask(para.MaxT,para.MaxD,para.MaxLen)
        self.Dropout=nn.Dropout(p=para.DropRate)
        self.RecPOITraEncLayer=nn.TransformerEncoderLayer(para.EmbDim,para.NHead,batch_first=True)
        self.PerPOITraEncLayer=nn.TransformerEncoderLayer(para.EmbDim,para.NHead,batch_first=True)
        self.RecPOITraEnc=nn.TransformerEncoder(self.RecPOITraEncLayer,para.LayerNum)
        self.PerPOITraEnc=nn.TransformerEncoder(self.PerPOITraEncLayer,para.LayerNum)
        self.PerCatTSExt=Attention(para.EmbDim)
        self.RecCatTSExt=Attention(para.EmbDim)
        self.PerCatTSAttFus=Attention(para.EmbDim)
        self.RecCatTSAttFus=Attention(para.EmbDim)
        self.PredictFC=nn.Linear(para.EmbDim,para.LocSize)

    def forward(self, inp, global_inp, is_dic, device):
        # to gpu
        for key in inp:
            if type(inp[key])!=list:
                inp[key]=inp[key].to(device)
        if len(inp['per_cat_s'])>0:
            cur_bat_cat=torch.tensor([inp['tg_cat'][:ind].tolist()+[0.0]*(inp['tg_cat'].shape[0]-1-ind) for ind in range(inp['tg_cat'].shape[0])]).to(device)
            per_cat_of_d=torch.cat(inp['per_cat_s'][-self.para.PerSloSize:],dim=0).unsqueeze(0).repeat(cur_bat_cat.shape[0],1).to(device)
            per_cat_seq=torch.cat([per_cat_of_d,cur_bat_cat],dim=1).to(device)
            per_cat_valLen=torch.tensor([per_cat_of_d.shape[1]+i for i in range(inp['tg_cat'].shape[0])]).to(device)
            cur_bat_poi=torch.tensor([inp['tg_poi'][:ind].tolist()+[0.0]*(inp['tg_poi'].shape[0]-1-ind) for ind in range(inp['tg_poi'].shape[0])]).to(device)
            per_poi_of_d=torch.cat(inp['per_poi_s'][-self.para.PerSloSize:],dim=0).unsqueeze(0).repeat(cur_bat_poi.shape[0],1).to(device)
            per_poi_seq=torch.cat([per_poi_of_d,cur_bat_poi],dim=1).to(device)
            per_poi_valLen=torch.tensor([per_poi_of_d.shape[1]+i for i in range(inp['tg_poi'].shape[0])]).to(device)
            if is_dic['POICat']=='Combine':
                if is_dic['is_Dropout']:
                    per_emb=self.Dropout(self.CatEmb(per_cat_seq.long())+self.POIEmb(per_poi_seq.long()))
                else:
                    per_emb=self.CatEmb(per_cat_seq.long())+self.POIEmb(per_poi_seq.long())
            elif is_dic['POICat']=='Split':
                if is_dic['is_Dropout']:
                    per_cat_emb=self.Dropout(self.CatEmb(per_cat_seq.long()))
                    per_poi_emb=self.Dropout(self.POIEmb(per_poi_seq.long()))
                else:
                    per_cat_emb=self.CatEmb(per_cat_seq.long())
                    per_poi_emb=self.POIEmb(per_poi_seq.long())
            per_mask_mat=(per_cat_seq==0).unsqueeze(1).repeat(1,per_cat_seq.shape[1],1)
            if is_dic['POICat']=='Combine':
                per_out=self.PerPOITraEnc(per_emb,mask=torch.repeat_interleave(per_mask_mat,self.para.NHead,dim=0))
            elif is_dic['POICat']=='Split':
                per_cat_out=self.PerCatTraEnc(per_cat_emb,mask=torch.repeat_interleave(per_mask_mat,self.para.NHead,dim=0))
                per_poi_out=self.PerPOITraEnc(per_poi_emb,mask=torch.repeat_interleave(per_mask_mat,self.para.NHead,dim=0))
        if is_dic['POICat']=='Combine':
            if is_dic['is_Dropout']:
                rec_emb=self.Dropout(self.CatEmb(inp['c_s'])+self.POIEmb(inp['p_s'])+self.PosEnc(inp['t_s'],device))
            else:
                rec_emb=self.CatEmb(inp['c_s'])+self.POIEmb(inp['p_s'])+self.PosEnc(inp['t_s'],device)
        elif is_dic['POICat']=='Split':
            if is_dic['is_Dropout']:
                rec_cat_emb=self.Dropout(self.CatEmb(inp['c_s'])+self.PosEnc(inp['t_s'],device))
                rec_poi_emb=self.Dropout(self.POIEmb(inp['p_s'])+self.PosEnc(inp['t_s'],device))
            else:
                rec_cat_emb=self.CatEmb(inp['c_s'])+self.PosEnc(inp['t_s'],device)
                rec_poi_emb=self.POIEmb(inp['p_s'])+self.PosEnc(inp['t_s'],device)
        rec_spa_tem_mat=self.STMask(inp['t_s'],inp['s_s'],inp['val_len'],device)
        if is_dic['POICat']=='Combine':
            rec_out=self.RecPOITraEnc(rec_emb,mask=torch.repeat_interleave(rec_spa_tem_mat,self.para.NHead,dim=0))
        elif is_dic['POICat']=='Split':
            rec_cat_out=self.RecCatTraEnc(rec_cat_emb,mask=torch.repeat_interleave(rec_spa_tem_mat,self.para.NHead,dim=0))
            rec_poi_out=self.RecPOITraEnc(rec_poi_emb,mask=torch.repeat_interleave(rec_spa_tem_mat,self.para.NHead,dim=0))
        if is_dic['is_TS_pref']:
            rec_ps=0
            if is_dic['POICat']=='Split':
                rec_out=rec_cat_out
            for bat_ind in range(rec_out.shape[0]):
                rec_ts_ps=0
                for win_ind in range(inp['ts_ran'].shape[1]):
                    start=inp['ts_ran'][bat_ind][win_ind][0]
                    end=inp['ts_ran'][bat_ind][win_ind][1]
                    if start!=end:
                        if type(rec_ts_ps)==int:
                            rec_ts_ps,att_w=self.RecCatTSExt(rec_out[bat_ind][start:end].unsqueeze(0))
                        else:
                            rec_ts_p,att_w=self.RecCatTSExt(rec_out[bat_ind][start:end].unsqueeze(0))
                            rec_ts_ps=torch.cat([rec_ts_ps,rec_ts_p],dim=0)
                rec_p,_=self.RecCatTSAttFus(rec_ts_ps.unsqueeze(0))
                if type(rec_ps)==int:
                    rec_ps=rec_p
                else:
                    rec_ps=torch.cat([rec_ps,rec_p],dim=0)
        if len(inp['per_cat_s'])>0:
            if is_dic['is_TS_pref']:
                c_num=[x.shape[0] for x in inp['per_cat_s'][-self.para.PerSloSize:]]
                cur_i,rang_ind=0,[]
                for num in c_num:
                    rang_ind.append([cur_i,cur_i+num])
                    cur_i=cur_i+num
                rang_ind=torch.cat([torch.tensor(rang_ind).unsqueeze(0).repeat(inp['tg_cat'].shape[0],1,1),torch.tensor([[rang_ind[-1][-1],rang_ind[-1][-1]+i] for i in range(inp['tg_cat'].shape[0])]).unsqueeze(1)],dim=1)
                per_ps=0
                if is_dic['POICat']=='Split':
                    per_out=per_cat_out
                for bat_ind in range(per_out.shape[0]):
                    per_ts_ps=0
                    for win_ind in range(rang_ind.shape[1]):
                        start=rang_ind[bat_ind][win_ind][0]
                        end=rang_ind[bat_ind][win_ind][1]
                        if start!=end:
                            if type(per_ts_ps)==int:
                                per_ts_ps,att_w=self.PerCatTSExt(per_out[bat_ind][start:end].unsqueeze(0))
                            else:
                                per_ts_p,att_w=self.PerCatTSExt(per_out[bat_ind][start:end].unsqueeze(0))
                                per_ts_ps=torch.cat([per_ts_ps,per_ts_p],dim=0)
                    per_p,_=self.PerCatTSAttFus(per_ts_ps.unsqueeze(0))
                    if type(per_ps)==int:
                        per_ps=per_p
                    else:
                        per_ps=torch.cat([per_ps,per_p],dim=0)#[B,D]
                delta_t=torch.abs((inp['tg_times']-inp['t_s'][-1][-inp['t_s'].shape[0]:])*self.para.TimSca)
                tem_dis_w=(torch.exp(-delta_t)).unsqueeze(-1)
                if is_dic['POICat']=='Combine':
                    per_rep=per_ps
                    rec_rep=rec_ps
                    if self.para.RecOnly:
                        enc_out=rec_rep
                    else:
                        enc_out=rec_rep*tem_dis_w+per_rep
                elif is_dic['POICat']=='Split':
                    cat_rep=rec_ps*tem_dis_w+per_ps #[B,D]
                    poi_rep=rec_poi_out[torch.arange(rec_cat_out.shape[0]),inp['val_len']-1]*tem_dis_w+per_poi_out[torch.arange(per_poi_out.shape[0]),per_poi_valLen-1]#[B,D]
            else:
                delta_t=torch.abs((inp['tg_times']-inp['t_s'][-1][-inp['t_s'].shape[0]:])*self.para.TimSca)
                tem_dis_w=(torch.exp(-delta_t)).unsqueeze(-1)
                if is_dic['POICat']=='Combine':
                    per_rep=per_out[torch.arange(per_out.shape[0]),per_poi_valLen-1]#[B,D]
                    rec_rep=rec_out[torch.arange(rec_out.shape[0]),inp['val_len']-1]
                    enc_out=rec_rep*tem_dis_w+per_rep
                elif is_dic['POICat']=='Split':
                    cat_rep=rec_cat_out[torch.arange(rec_cat_out.shape[0]),inp['val_len']-1]*tem_dis_w+per_cat_out[torch.arange(per_cat_out.shape[0]),per_cat_valLen-1]#[B,D]
                    poi_rep=rec_poi_out[torch.arange(rec_poi_out.shape[0]),inp['val_len']-1]*tem_dis_w+per_poi_out[torch.arange(per_poi_out.shape[0]),per_poi_valLen-1]#[B,D]

        else:
            if is_dic['is_TS_pref']:
                if is_dic['POICat']=='Combine':
                    rec_rep=rec_ps
                elif is_dic['POICat']=='Split':
                    cat_rep=rec_ps
            else:
                if is_dic['POICat']=='Combine':
                    rec_rep=rec_out[torch.arange(rec_out.shape[0]),inp['val_len']-1]
                elif is_dic['POICat']=='Split':
                    cat_rep=rec_cat_out[torch.arange(rec_cat_out.shape[0]),inp['val_len']-1]
            if is_dic['POICat']=='Combine':
                enc_out=rec_rep
            elif is_dic['POICat']=='Split':
                poi_rep=rec_poi_out[torch.arange(rec_poi_out.shape[0]),inp['val_len']-1]
        if is_dic['POICat']=='Split':
            if self.para.Fusion=='Add':
                enc_out=(cat_rep+poi_rep)
            elif self.para.Fusion=='Concat':
                enc_out=torch.cat([cat_rep,poi_rep],dim=1)
                enc_out=self.ConFC(enc_out)
            elif self.para.Fusion=='Att':
                enc_out,att_w=self.PrefHisAttFus(torch.cat([cat_rep.unsqueeze(1),poi_rep.unsqueeze(1)],dim=1))
                enc_out=enc_out
        mat_res=self.PredictFC(enc_out)
        if is_dic['is_train']:
            return mat_res,enc_out
        else:
            sort_res=mat_res.sort(-1,True).indices
            return sort_res


class PrefSimMat(nn.Module):
    def __init__(self):
        super(PrefSimMat, self).__init__()
        self.SoftMax=nn.Softmax(dim=-1)

    def forward(self, p_u, mode):
        r1=torch.repeat_interleave(p_u,p_u.shape[0],dim=0)
        r2=p_u.repeat(p_u.shape[0],1)
        if mode=='EucDis':
            sim_mat=torch.norm(r1-r2,p=2,dim=1).reshape(p_u.shape[0],-1)
            sim_mat=1-F.normalize(sim_mat,dim=-1)
        elif mode=='CosSim':
            cos=nn.CosineSimilarity(dim=1)
            sim_mat=torch.abs(cos(r1,r2)).reshape(p_u.shape[0],-1)

        return Variable(sim_mat,requires_grad=False)

class PolicyNet(nn.Module):
    def __init__(self, para):
        super(PolicyNet, self).__init__()
        self.para=para
        self.DataNumEmb=nn.Embedding(para.CatNDataN, para.PolEmbDim)
        self.PSMask=PrefSimMat()
        self.Dropout=nn.Dropout(p=para.PolDropRate)
        self.TraEncLayer=nn.TransformerEncoderLayer(para.PolEmbDim,para.PolNHead,batch_first=True)
        self.TraEnc=nn.TransformerEncoder(self.TraEncLayer,para.PolLayerNum)
        self.FC=nn.Linear(para.PolEmbDim,para.PolCluNum)
        self.SoftMax=nn.Softmax(dim=-1)

    def forward(self, inp):
        data_num_emb=self.DataNumEmb(inp['d_n'])
        data_num_emb=self.Dropout(data_num_emb)
        if self.para.PolDisMode=='RBF':
            sim_mat=inp['a_m']
        else:
            sim_mat=self.PSMask(inp['p_u'],self.para.PolDisMode)
        tran_out=self.TraEnc(data_num_emb.unsqueeze(0),mask=sim_mat)
        fc_out=self.FC(tran_out)
        
        return fc_out.squeeze(0)