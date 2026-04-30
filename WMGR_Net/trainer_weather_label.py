import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F


def train(train_config, model, dataloader, loss_function, weather_loss_function, optimizer, scheduler=None, scaler=None):
    # set model train mode
    model.train()

    losses = AverageMeter()
    global_losses = AverageMeter()
    local_losses = AverageMeter()
    semantic_losses = AverageMeter()
    # 新增：监控天气 Loss
    losses_w = AverageMeter()
    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    # for loop over one epoch
    for query, reference, ids, weather_labels in bar:

        if scaler:
            with autocast():

                # data (batches) to device
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                weather_labels = weather_labels.to(train_config.device)
                # Forward pass
                output1, output2, weather_logits = model(query, reference, return_local=True)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:

                    loss = loss_function(output1, output2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(output1, output2, model.logit_scale.exp())

                # 2. 计算天气 Loss (CrossEntropy)
                loss_w = 0.0
                if weather_logits is not None:
                    # lambda_weather 系数，可以写死 0.1 或从 config 传入
                    loss_w = weather_loss_function(weather_logits, weather_labels)

                # 总 Loss
                lambda_w = 0.15  # 建议权重
                loss = loss + lambda_w * loss_w
                val_w = loss_w.item() if isinstance(loss_w, torch.Tensor) else loss_w
                losses_w.update(val_w)
                losses.update(loss.item())

                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected at step {step}, skipping...")
                    continue

                # 记录详细损失（如果可用）
                if hasattr(loss_function, 'last_global_loss'):
                    global_losses.update(loss_function.last_global_loss)
                if hasattr(loss_function, 'last_local_loss'):
                    local_losses.update(loss_function.last_local_loss)
                if hasattr(loss_function, 'last_semantic_loss'):
                    semantic_losses.update(loss_function.last_semantic_loss)

            scaler.scale(loss).backward()

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:

            # data (batches) to device
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_w": "{:.4f}".format(losses_w.avg),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            # 添加详细损失信息（如果可用）
            if hasattr(loss_function, 'last_global_loss') and global_losses.count > 0:
                monitor.update({
                    "global": "{:.3f}".format(global_losses.avg),
                    "local": "{:.3f}".format(local_losses.avg) if local_losses.count > 0 else "0.000",
                    "semantic": "{:.3f}".format(semantic_losses.avg) if semantic_losses.count > 0 else "0.000"
                })
            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []
    with torch.no_grad():

        for img, ids in bar:

            ids_list.append(ids)

            with autocast():

                img = img.to(train_config.device)
                output = model(img, return_local=True)
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(output[0], dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

    if train_config.verbose:
        bar.close()

    return img_features, ids_list