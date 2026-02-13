import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from src.model.CNN_Attention import cnn_PatchTST, cnn_MLLA, Conv_att_simple_mlp
from src.model.Channel_MLP import Channel_mlp_CNN
from src.model.PatchTSTsingle import PatchTST_single_backbone
from src.model.MLLA_new import channel_MLLA
from src.model.Ablation_Transformer import TemporalTransformer
from src.model.domain_head import DomainHead
from src.model.grl import grad_reverse
from src.loss.loss import SimCLRLoss
from src.loss.CDA_loss import CDALoss


class MultiModel_PL(pl.LightningModule):
    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_fea = False
        self.channel_projection_matrix = [[None] * len(self.cfg.data_cfg_list)][0]
        self.channel_interpolate = np.load('channel_interpolate.npy').astype(int)
        self.uni_channelname = self.cfg.model.MLLA.uni_channels

        if cfg.model.encoder == 'cnn':
            self.cnn_encoder = Conv_att_simple_mlp(
                cfg.model.cnn.n_timeFilters,
                cfg.model.cnn.timeFilterLen,
                cfg.model.cnn.n_msFilters,
                cfg.model.cnn.msFilter_timeLen,
                cfg.model.cnn.n_channs,
                cfg.model.cnn.dilation_array,
                cfg.model.cnn.seg_att,
                cfg.model.cnn.avgPoolLen,
                cfg.model.cnn.timeSmootherLen,
                cfg.model.cnn.multiFact,
                cfg.model.cnn.stratified,
                cfg.model.cnn.activ,
                cfg.model.cnn.temp,
                cfg.model.cnn.saveFea,
                cfg.model.cnn.has_att,
                cfg.model.cnn.extract_mode,
                cfg.model.cnn.global_att,
                c_mlps=[Channel_mlp_CNN(cfg_i.n_channs, cfg.model.cnn.n_channs) for cfg_i in cfg.data_cfg_list],
            )

        if cfg.model.encoder == 'TST_single':
            self.c_mlps = [Channel_mlp_CNN(cfg_i.n_channs, cfg.model.TST_single.cnn.n_channs) for cfg_i in cfg.data_cfg_list]
            self.patchTST = PatchTST_single_backbone(
                c_in=1,
                context_window=cfg.data_0.timeLen * cfg.data_0.fs,
                patch_len=cfg.model.TST_single.patch_len,
                stride=cfg.model.TST_single.patch_stride,
                d_model=cfg.model.TST_single.cnn.n_timeFilters,
                n_heads=cfg.model.TST_single.n_heads,
            )
            self.cnn_encoder = cnn_PatchTST(
                cfg.model.TST_single.cnn.n_timeFilters,
                cfg.model.TST_single.cnn.timeFilterLen,
                cfg.model.TST_single.cnn.n_msFilters,
                cfg.model.TST_single.cnn.msFilter_timeLen,
                cfg.model.TST_single.cnn.n_channs,
                cfg.model.TST_single.cnn.dilation_array,
                cfg.model.TST_single.cnn.seg_att,
                cfg.model.TST_single.cnn.avgPoolLen,
                cfg.model.TST_single.cnn.timeSmootherLen,
                cfg.model.TST_single.cnn.multiFact,
                cfg.model.TST_single.cnn.stratified,
                cfg.model.TST_single.cnn.activ,
                cfg.model.TST_single.cnn.temp,
                cfg.model.TST_single.cnn.saveFea,
                cfg.model.TST_single.cnn.has_att,
                cfg.model.TST_single.cnn.extract_mode,
                cfg.model.TST_single.cnn.global_att,
            )

        if cfg.model.encoder == 'MLLA':
            self.uni_mlp = Channel_mlp_CNN(len(self.uni_channelname), cfg.model.MLLA.cnn.n_channs)
            self.MLLA = channel_MLLA(
                context_window=cfg.data_0.timeLen * cfg.data_0.fs,
                patch_size=cfg.model.MLLA.patch_size,
                hidden_dim=cfg.model.MLLA.hidden_dim,
                out_dim=cfg.model.MLLA.out_dim,
                depth=cfg.model.MLLA.depth,
                patch_stride=cfg.model.MLLA.patch_stride,
                n_heads=cfg.model.MLLA.n_heads,
            )
            self.cnn_encoder = cnn_MLLA(
                cfg.model.MLLA.cnn.n_timeFilters,
                cfg.model.MLLA.cnn.timeFilterLen,
                cfg.model.MLLA.cnn.n_msFilters,
                cfg.model.MLLA.cnn.msFilter_timeLen,
                cfg.model.MLLA.cnn.n_channs,
                cfg.model.MLLA.cnn.dilation_array,
                cfg.model.MLLA.cnn.seg_att,
                cfg.model.MLLA.cnn.avgPoolLen,
                cfg.model.MLLA.cnn.timeSmootherLen,
                cfg.model.MLLA.cnn.multiFact,
                cfg.model.MLLA.cnn.stratified,
                cfg.model.MLLA.cnn.activ,
                cfg.model.MLLA.cnn.temp,
                cfg.model.MLLA.cnn.saveFea,
                cfg.model.MLLA.cnn.has_att,
                cfg.model.MLLA.cnn.extract_mode,
                cfg.model.MLLA.cnn.global_att,
            )

        if cfg.model.encoder == 'Transformer':
            self.transformer_encoder = TemporalTransformer(
                n_chann=len(self.uni_channelname),
                dim=cfg.model.Transformer.dim,
                dim_out=cfg.model.Transformer.out_dim,
                n_heads=cfg.model.Transformer.n_heads,
            )
            self.uni_mlp = Channel_mlp_CNN(len(self.uni_channelname), cfg.model.MLLA.cnn.n_channs)
            self.cnn_encoder = cnn_MLLA(
                cfg.model.MLLA.cnn.n_timeFilters,
                cfg.model.MLLA.cnn.timeFilterLen,
                cfg.model.MLLA.cnn.n_msFilters,
                cfg.model.MLLA.cnn.msFilter_timeLen,
                cfg.model.MLLA.cnn.n_channs,
                cfg.model.MLLA.cnn.dilation_array,
                cfg.model.MLLA.cnn.seg_att,
                cfg.model.MLLA.cnn.avgPoolLen,
                cfg.model.MLLA.cnn.timeSmootherLen,
                cfg.model.MLLA.cnn.multiFact,
                cfg.model.MLLA.cnn.stratified,
                cfg.model.MLLA.cnn.activ,
                cfg.model.MLLA.cnn.temp,
                cfg.model.MLLA.cnn.saveFea,
                cfg.model.MLLA.cnn.has_att,
                cfg.model.MLLA.cnn.extract_mode,
                cfg.model.MLLA.cnn.global_att,
            )

        self.clisa_loss = SimCLRLoss(cfg.train.loss.temp)
        self.cda_loss = CDALoss(cfg)

        self.n_datasets = len(self.cfg.data_cfg_list)
        self.n_envs = max(1, int(getattr(getattr(cfg.train, 'env', {}), 'num_envs', 1)))
        self.domain_label_mode = str(getattr(getattr(cfg.train, 'env', {}), 'domain_label_mode', 'combined'))

        if self.domain_label_mode == 'dataset':
            self.domain_out_dim = self.n_datasets
        elif self.domain_label_mode == 'env':
            self.domain_out_dim = self.n_envs
        else:
            self.domain_out_dim = self.n_datasets * self.n_envs

        domain_hidden = int(getattr(cfg.train.loss, 'domain_hidden_dim', 128))
        domain_dropout = float(getattr(cfg.train.loss, 'domain_dropout', 0.1))
        self.domain_head = DomainHead(self.domain_out_dim, hidden_dim=domain_hidden, dropout=domain_dropout)
        self.dataset_label_heads = nn.ModuleList([nn.LazyLinear(int(cfg_i.n_class)) for cfg_i in cfg.data_cfg_list])

        self.adv_factor = float(getattr(cfg.train.loss, 'adv_factor', 0.0))
        self.irm_factor = float(getattr(cfg.train.loss, 'irm_factor', 0.0))
        self.invaug_factor = float(getattr(cfg.train.loss, 'invaug_factor', 0.0))
        self.invaug_enable = bool(getattr(cfg.train.loss, 'invaug_enable', True))
        self.invaug_stopgrad = bool(getattr(cfg.train.loss, 'invaug_stopgrad', True))
        self.adv_grl_lambda = float(getattr(cfg.train.loss, 'adv_grl_lambda', 1.0))

        self.adv_warmup_epochs = int(getattr(getattr(cfg.train, 'adv', {}), 'warmup_epochs', 0))
        self.adv_ramp_epochs = int(getattr(getattr(cfg.train, 'adv', {}), 'ramp_epochs', 0))
        self.irm_warmup_epochs = int(getattr(getattr(cfg.train, 'irm', {}), 'warmup_epochs', 0))
        self.irm_ramp_epochs = int(getattr(getattr(cfg.train, 'irm', {}), 'ramp_epochs', 0))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.wd)
        return {'optimizer': optimizer}

    def _scheduled_weight(self, target, warmup_epochs, ramp_epochs):
        if target <= 0:
            return 0.0
        epoch = int(self.current_epoch)
        if epoch < warmup_epochs:
            return 0.0
        if ramp_epochs <= 0:
            return target
        progress = float(epoch - warmup_epochs + 1) / float(ramp_epochs)
        return target * min(1.0, max(0.0, progress))

    def _feature_to_embedding(self, fea):
        if fea.dim() > 2:
            return fea.flatten(start_dim=1)
        return fea

    def _irm_penalty(self, logits, labels):
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits * scale, labels)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def _unpack_batch(self, batch):
        def _normalize_dataset_tensor_list(field_list):
            normalized = []
            for item in field_list:
                if not torch.is_tensor(item):
                    raise TypeError("Expected tensor field in collated batch.")
                if item.dim() >= 2:
                    # Merge outer DataLoader batch dim and inner sampler dim.
                    item = item.flatten(0, 1)
                else:
                    item = item.reshape(-1)
                normalized.append(item)
            return normalized

        if isinstance(batch, dict):
            x_list = _normalize_dataset_tensor_list(batch['x'])
            y_list = [x.long() for x in _normalize_dataset_tensor_list(batch['y'])]
            x_a_list = _normalize_dataset_tensor_list(batch.get('x_a', batch['x']))
            x_b_list = _normalize_dataset_tensor_list(batch.get('x_b', batch['x']))
            env_id_list = [x.long() for x in _normalize_dataset_tensor_list(batch.get('env_id', []))]
            env_label_list = [x.long() for x in _normalize_dataset_tensor_list(batch.get('env_label', []))]
            return x_list, y_list, x_a_list, x_b_list, env_id_list, env_label_list

        # Backward compatibility.
        x_list, y_list = batch
        x_list = _normalize_dataset_tensor_list(x_list)
        y_list = [x.long() for x in _normalize_dataset_tensor_list(y_list)]
        x_a_list = x_list
        x_b_list = x_list
        env_id_list = [torch.zeros(x_i.shape[0], dtype=torch.long, device=x_i.device) for x_i in x_list]
        env_label_list = [torch.full((x_i.shape[0],), d, dtype=torch.long, device=x_i.device) for d, x_i in enumerate(x_list)]
        return x_list, y_list, x_a_list, x_b_list, env_id_list, env_label_list

    def forward(self, x, dataset=0, returnMLLAout=False):
        if self.cfg.model.encoder == 'cnn':
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x, dataset)
            return x

        if self.cfg.model.encoder == 'TST_single':
            x = x.squeeze(1)
            x = self.patchTST(x)
            x = torch.permute(x, (0, 2, 1, 3))
            x = self.c_mlps[dataset](x)
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x)
            return x

        if self.cfg.model.encoder == 'MLLA':
            x = self.MLLA(x)
            if returnMLLAout:
                mllaout = x
            x = torch.permute(x, (0, 3, 1, 2))
            x = self.uni_mlp(x)
            fea_cov = x
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x)
            if returnMLLAout:
                return x, fea_cov, mllaout
            return x, fea_cov

        if self.cfg.model.encoder == 'Transformer':
            x = self.transformer_encoder(x)
            x = self.uni_mlp(x)
            fea_cov = x
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x)
            return x, fea_cov

    def _forward_features(self, x, dataset):
        out = self.forward(x, dataset)
        if isinstance(out, tuple):
            if len(out) >= 2:
                return out[0], out[1]
            return out[0], None
        return out, None

    def training_step(self, batch, batch_idx):
        x_list, y_list, x_a_list, x_b_list, env_id_list, env_label_list = self._unpack_batch(batch)
        n_dataset = len(x_list)

        x_list = [self.channel_project(x_list[i], self.cfg.data_cfg_list[i].channels) for i in range(n_dataset)]
        x_a_list = [self.channel_project(x_a_list[i], self.cfg.data_cfg_list[i].channels) for i in range(n_dataset)]
        x_b_list = [self.channel_project(x_b_list[i], self.cfg.data_cfg_list[i].channels) for i in range(n_dataset)]

        z_list, cov_list = [], []
        for dataset in range(n_dataset):
            z_i, cov_i = self._forward_features(x_list[dataset], dataset)
            z_list.append(z_i)
            if cov_i is not None:
                cov_list.append(cov_i)

        device = z_list[0].device
        loss_total = torch.zeros((), device=device)
        loss_isa = torch.zeros((), device=device)
        loss_cda = torch.zeros((), device=device)
        loss_adv = torch.zeros((), device=device)
        loss_irm = torch.zeros((), device=device)
        loss_inv = torch.zeros((), device=device)

        if self.cfg.train.loss.clisa_loss:
            for dataset, z_i in enumerate(z_list):
                emb_i = self._feature_to_embedding(z_i)
                clisa_loss_i, _, _, _ = self.clisa_loss(emb_i)
                loss_isa = loss_isa + clisa_loss_i
                self.log(
                    f'loss_clisa_{self.cfg.data_cfg_list[dataset].dataset_name}/train',
                    clisa_loss_i,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            loss_total = loss_total + loss_isa

        if self.cfg.train.loss.CDA_loss and len(cov_list) > 0:
            loss_cda = self.cda_loss(cov_list) * self.cfg.train.loss.CDA_factor
            loss_total = loss_total + loss_cda

        adv_weight = self._scheduled_weight(self.adv_factor, self.adv_warmup_epochs, self.adv_ramp_epochs)
        if adv_weight > 0 and len(env_label_list) == n_dataset:
            domain_losses = []
            domain_accs = []
            for dataset, z_i in enumerate(z_list):
                emb_i = self._feature_to_embedding(z_i)
                env_label = env_label_list[dataset].to(emb_i.device).long()
                logits_env = self.domain_head(grad_reverse(emb_i, self.adv_grl_lambda))
                dom_loss_i = F.cross_entropy(logits_env, env_label)
                domain_losses.append(dom_loss_i)
                dom_acc_i = (torch.argmax(logits_env, dim=1) == env_label).float().mean()
                domain_accs.append(dom_acc_i)
                self.log(
                    f'domain_acc_{self.cfg.data_cfg_list[dataset].dataset_name}/train',
                    dom_acc_i,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
            loss_adv = torch.stack(domain_losses).mean()
            loss_total = loss_total + adv_weight * loss_adv
            self.log('domain_acc/train', torch.stack(domain_accs).mean(), on_step=False, on_epoch=True, prog_bar=False)

        irm_weight = self._scheduled_weight(self.irm_factor, self.irm_warmup_epochs, self.irm_ramp_epochs)
        if irm_weight > 0 and len(env_id_list) == n_dataset:
            penalties = []
            for dataset, z_i in enumerate(z_list):
                emb_i = self._feature_to_embedding(z_i)
                labels = y_list[dataset].to(emb_i.device).long()
                env_ids = env_id_list[dataset].to(emb_i.device).long()
                logits_raw = self.dataset_label_heads[dataset](emb_i)

                unique_envs = torch.unique(env_ids)
                for env_id in unique_envs:
                    mask = env_ids == env_id
                    if int(mask.sum().item()) == 0:
                        continue
                    ce_env = F.cross_entropy(logits_raw[mask], labels[mask])
                    penalty_env = self._irm_penalty(logits_raw[mask], labels[mask])
                    penalties.append(penalty_env)
                    self.log(
                        f'irm_ce_{self.cfg.data_cfg_list[dataset].dataset_name}_e{int(env_id.item())}/train',
                        ce_env,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    self.log(
                        f'irm_penalty_{self.cfg.data_cfg_list[dataset].dataset_name}_e{int(env_id.item())}/train',
                        penalty_env,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
            if len(penalties) > 0:
                loss_irm = torch.stack(penalties).mean()
                loss_total = loss_total + irm_weight * loss_irm

        if self.invaug_enable and self.invaug_factor > 0:
            inv_losses = []
            for dataset in range(n_dataset):
                z_a, _ = self._forward_features(x_a_list[dataset], dataset)
                z_b, _ = self._forward_features(x_b_list[dataset], dataset)
                emb_a = self._feature_to_embedding(z_a)
                emb_b = self._feature_to_embedding(z_b)
                if self.invaug_stopgrad:
                    emb_b = emb_b.detach()
                inv_losses.append(F.mse_loss(emb_a, emb_b))
            if len(inv_losses) > 0:
                loss_inv = torch.stack(inv_losses).mean()
                loss_total = loss_total + self.invaug_factor * loss_inv

        self.log_dict(
            {
                'loss_total/train': loss_total,
                'loss_isa/train': loss_isa,
                'loss_cda/train': loss_cda,
                'loss_adv/train': loss_adv,
                'loss_irm/train': loss_irm,
                'loss_invaug/train': loss_inv,
                'weight_adv/train': torch.tensor(adv_weight, device=device),
                'weight_irm/train': torch.tensor(irm_weight, device=device),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss_total

    def validation_step(self, batch, batch_idx):
        loss = 0
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        x = self.channel_project(x, self.cfg.data_val.channels)
        out = self.forward(x, 0, returnMLLAout=True)
        if isinstance(out, tuple):
            if len(out) >= 3:
                return out[0], out[1], out[2]
            if len(out) == 2:
                return out[0], out[1], out[0]
            return out[0], out[0], out[0]
        return out, out, out

    def channel_project(self, data, cha_source):
        device = data.device
        batch_size, _, n_channel_source, n_timepoint = data.shape
        n_channel_standard = len(self.uni_channelname)

        source_ch_map = {name.upper(): idx for idx, name in enumerate(cha_source)}
        result = torch.zeros(
            (batch_size, 1, n_channel_standard, n_timepoint),
            device=device,
            dtype=data.dtype,
        )

        for std_idx, std_name in enumerate(self.uni_channelname):
            std_name_upper = std_name.upper()
            if std_name_upper in source_ch_map:
                src_idx = source_ch_map[std_name_upper]
                result[:, :, std_idx] = data[:, :, src_idx]
                continue

            neighbor_std_indices = self.channel_interpolate[std_idx]
            valid_src_indices = []
            for neighbor_std_idx in neighbor_std_indices:
                neighbor_std_name = self.uni_channelname[neighbor_std_idx.item()].upper()
                if neighbor_std_name in source_ch_map:
                    valid_src_indices.append(source_ch_map[neighbor_std_name])
                    if len(valid_src_indices) == 3:
                        break

            if len(valid_src_indices) > 0:
                neighbor_data = data[:, :, valid_src_indices, :]
                interpolated = neighbor_data.mean(dim=2)
                result[:, :, std_idx] = interpolated
            else:
                print(f"Channel {std_name} has no available neighbors, filled with zeros")
        return result
