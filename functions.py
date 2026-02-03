import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import functools
import torch.nn.functional as F
from PIL import Image

def check_cuda_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        check = kwargs.pop("check_cuda_memory", False)
        if check and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            result = func(*args, **kwargs)
            cuda_memory_usage = round(torch.cuda.memory_allocated() / 1024**2, 1)
            print(f"CUDA Memory Usage: {cuda_memory_usage} MB")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num

@check_cuda_memory
def get_epoch_loss(model, device, dataloader, criterion, optimizer = None):
    N = len(dataloader.dataset)
    r_loss = 0; correct_e = 0
    for x_batch, y_batch in tqdm(dataloader, leave = False):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_out = model(x_batch) # bs, 10
        loss = criterion(y_out, y_batch)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_b = loss.item() * x_batch.shape[0]
        r_loss += loss_b
        pred = torch.argmax(y_out, dim = 1)
        correct_b = torch.sum(pred == y_batch).item()
        correct_e += correct_b
    loss_e = r_loss / N
    accuracy = correct_e / N
    return loss_e, accuracy, correct_e


def Train(model, epoch, device, Train_DL, val_DL, criterion, optimizer, base_model_path, code, **kwargs):

    train_start_time = time.time()

    if "scheduler" in kwargs:
        scheduler = kwargs["scheduler"]
    else:
        scheduler = None
    best_val_loss = 9999
    loss_history = {"Train" : [],
                    "Validation" : [],
                    "Test" : []}
    acc_history = {"Train" : [],
                    "Validation" : [],
                    "Test" : []}
    
    for ep in range(epoch):
        ep_start_time = time.time()
        model.train()
        train_loss, train_acc, _ =get_epoch_loss(model, device, Train_DL, criterion, optimizer, check_cuda_memory=True)
        loss_history["Train"].extend([train_loss])
        acc_history["Train"].extend([train_acc])

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _ =get_epoch_loss(model, device, val_DL, criterion, optimizer = None)
            loss_history["Validation"].extend([val_loss])
            acc_history["Validation"].extend([val_acc])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                file_name = f"{code}_{time.strftime('%Y%m%d')}_{ep+1}.pt"
                save_model_path = os.path.join(base_model_path, file_name)
                torch.save({"epoch": ep+1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,}, save_model_path)
                print(f"Model saved at epoch {ep+1} with validation loss {round(val_loss,3)} in {save_model_path}")
        if scheduler is not None:
            scheduler.step()
        print(f"EPOCH: {ep+1} /LR: {scheduler.get_last_lr()[0]:.6f} / EPOCH Duration: {round(time.time() - ep_start_time, 2)} sec")
        print(f"train loss: {round(train_loss,3)} / Validation loss: {round(val_loss,3)} / Validation accuracy: {round(val_acc,3)}")
    print(f"Total Training Time: {round(time.time() - train_start_time, 2)} sec")
    return loss_history

def Test(model, device, dataloader, criterion):
    model.eval()
    with torch.no_grad():
       test_loss, test_acc, t_correct = get_epoch_loss(model, device, dataloader, criterion, optimizer = None)
       print(f"Test Loss: {round(test_loss,5)} / Test Accuracy: {round(test_acc,5)}")
       print(f"Correct {t_correct} out of {len(dataloader.dataset)}")
    return test_acc

def plot_loss(loss_history):
    plt.plot(loss_history["Train"], label="Train Loss")
    plt.plot(loss_history["Validation"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def visualize_wrong(model, dataloader, device, n_of_show = 10):
    model.eval()
    wrong_ex = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device) # shape = 32,1
            output = model(imgs) #32,10
            preds = torch.argmax(output, dim = 1) # 32

            wrong_indices = (preds != labels).nonzero().flatten()

            for idx in wrong_indices:
                if len(wrong_ex) >= n_of_show:
                    break

                img = imgs[idx].cpu()
                label = labels[idx].item()
                pred = preds[idx].item()
                wrong_ex.append((img, pred, label))
            if len(wrong_ex) >= n_of_show:
                break
    
    import math
    ncols = 5
    nrows = math.ceil(len(wrong_ex) / ncols)   
    
    plt.figure(figsize=(15, 3 * nrows))
    for i, (img, pred, label) in enumerate(wrong_ex):
        plt.subplot(nrows, ncols, i+1) # nrows행 ncols열 (최대 n_of_show개 기준)
        
        # Tensor (C, H, W) -> Numpy (H, W, C) 변환
        # 정규화(Normalize)를 했다면 여기서 역정규화(Un-normalize) 코드가 필요할 수 있습니다.
        img_np = img.permute(1, 2, 0).numpy()
        
        # 이미지가 흑백(1채널)인 경우 shape 보정 및 colormap 설정
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)
            plt.imshow(img_np, cmap='gray')
        else:
            # Min-Max Scaling으로 0~1 사이로 값 보정 (시각화를 위해)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            plt.imshow(img_np)
            
        plt.title(f"Pred: {pred} | GT: {label}", color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def Test_plot(model, test_DL, device):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(device)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to("cpu")

    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1,2,0).squeeze(), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color = "g" if pred_class==true_class else "r")     

def im_plot(DL):
    x_batch, y_batch = next(iter(DL))
    plt.figure(figsize=(8,4))

    for idx in range(6):
        im = x_batch[idx]
        im = im-im.min() # for imshow clipping
        im = im/im.max() # for imshow clipping

        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(im.permute(1,2,0).squeeze())
        true_class = DL.dataset.classes[y_batch[idx]]
        plt.title(true_class, color = "g") 
    print(f"x_batch size = {x_batch.shape}")

def get_feature_maps(model, input_image, layer_names, transform=None):
    """
    이미지와 Transform을 받아 전처리 후 Feature Map을 추출하는 함수
    
    Args:
        model: PyTorch 모델 (GPU에 있어도 됨)
        input_image: 입력 이미지 (PIL Image, numpy array, 혹은 Tensor)
        layer_names: 추출할 레이어 이름 리스트 (예: ['conv1', 'layer1'])
        transform: 적용할 torchvision.transforms (없으면 None)
        
    Returns:
        features: {레이어이름: 텐서} 형태의 딕셔너리
    """
    features = {}
    handles = []
    
    def get_hook(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    modules = dict(model.named_modules())
    
    for name in layer_names:
        if name in modules:
            handle = modules[name].register_forward_hook(get_hook(name))
            handles.append(handle)
        else:
            print(f"경고: '{name}' 레이어를 찾을 수 없습니다.")

    if transform is not None:
        input_tensor = transform(input_image)
    else:
        input_tensor = input_image

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)


    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)


    model.eval()
    with torch.no_grad():
        model(input_tensor)

    for handle in handles:
        handle.remove()

    return features

def plot_feature_maps(feature_maps, layer_names, num_cols=8, overlay=False, input_image=None, alpha=0.5, colormap='jet'):
    """
    특정 레이어의 Feature Map을 시각화하는 함수
    
    Args:
        input_image: 원본 입력 이미지 (PIL Image 또는 numpy array)
        feature_maps: {레이어이름: 텐서} 형태의 딕셔너리
        layer_name: 시각화할 레이어 이름
        num_cols: 한 행에 표시할 Feature Map 개수
    """
    if overlay and input_image is not None:
        if input_image.dim() == 4:
            input_image = input_image.squeeze(0)
        img_np = input_image.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        h, w, _ = img_np.shape

    for layer_name in layer_names:
        if layer_name not in feature_maps:
            print(f" '{layer_name}' 레이어의 Feature Map을 찾을 수 없습니다.")
            continue
        
        if overlay:
            fmap = feature_maps[layer_name]  # (N, C, H, W)
            heatmap = torch.mean(fmap, dim=1, keepdim=True)
            heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
            heatmap = heatmap.squeeze().cpu().numpy() # [H, W]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            # 3. 그리기
            plt.figure(figsize=(6, 6))
            
            # 배경: 원본 이미지
            plt.imshow(img_np)
            
            # 오버레이: 히트맵 (alpha로 투명도 조절)
            plt.imshow(heatmap, cmap=colormap, alpha=alpha)
            
            plt.title(f"Layer: {layer_name} Overlay")
            plt.axis('off')
            plt.show()

        else:
            fmap = feature_maps[layer_name].squeeze(0)  # (C, H, W)
            num_feature_maps = fmap.shape[0]
            num_rows = (num_feature_maps + num_cols - 1) // num_cols

            plt.figure(figsize=(num_cols * 2, num_rows * 2))
            for i in range(num_feature_maps):
                plt.subplot(num_rows, num_cols, i + 1)
                plt.imshow(fmap[i].cpu(), cmap='viridis')
                plt.axis('off')
                plt.title(f'FM {i+1}')
            plt.suptitle(f'Feature Maps from Layer: {layer_name}', fontsize=16)
            plt.tight_layout()
            plt.show()