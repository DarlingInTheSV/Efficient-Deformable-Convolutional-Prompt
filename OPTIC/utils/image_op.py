import torch
from torchvision import transforms
from PIL import Image
import numpy as np
def save_prompt_img(img, path, is_origin=True, is_adapt=False, max_val=None, min_val=None):
    img_tensor = img.cpu()
    if not is_origin:
        # 最大最小值归一化
        normalized_tensor = img_tensor * (max_val - min_val) + min_val
        # rgb_tensor = normalized_tensor.byte()
        rgb_tensor = (normalized_tensor * 5).byte()  # 进一步扩大增加的和减少的差距， *2能看出主要改变量都在OC/OD上，*5 也表现出了，但还有部分空白，* 10 包括其背景等其他的细节

        # percentile_value = torch.quantile(img_tensor.flatten(), 0.5)
        # # 小于阈值的元素置为 0
        # img_tensor[img_tensor < percentile_value] = img_tensor.flatten().min()
        # rgb_tensor = img_tensor.byte()
    else:
        if is_adapt:
            normalized_tensor = img_tensor.clamp(0, 1)
            # normalized_tensor = img_tensor * (max_val - min_val) + min_val
        else:
            normalized_tensor = img_tensor
        rgb_tensor = (normalized_tensor * 255).byte()

    # 调整数据范围到 [0, 255]


    # 转换为 numpy 数组
    rgb_array = rgb_tensor.numpy()

    # 将通道维度从 (3, 512, 512) 转换为 (512, 512, 3)
    rgb_array = np.transpose(rgb_array, (1, 2, 0))

    # 将 numpy 数组转换为 PIL 图像
    rgb_image = Image.fromarray(rgb_array)

    # 保存图像
    rgb_image.save(path)

def save_to_png(img, path):

    # 定义转换：从张量到 PIL 图像
    to_pil_image = transforms.ToPILImage()

    # 将张量转换为 PIL 图像
    pil_image = to_pil_image(img)

    # 保存 PIL 图像为 PNG 文件
    output_path = path
    pil_image.save(output_path)
    # 输出保存成功的信息
    print(f"Image saved to {output_path}")

def save_label_mask(logits, path, gt=False):
    # Step 1: 选择通道 - 对于二分类任务，选择概率较高的那个通道
    # 使用 argmax 获取每个像素点的预测类别


    if not gt:
        pred = torch.sigmoid(logits[0])
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        predicted_classes_np = pred.detach().cpu().numpy().astype(np.uint8)
    else:
        predicted_classes_np = logits[0].detach().cpu().numpy().astype(np.uint8)

    # 提取两个通道的分类结果
    channel1 = predicted_classes_np[0]
    channel2 = predicted_classes_np[1]

    # 创建一个空的灰度图
    gray_image = np.zeros((512, 512), dtype=np.uint8)

    # (0, 0) -> 白色 (255)
    gray_image[(channel1 == 0) & (channel2 == 0)] = 255

    # (1, 0) -> 灰色 (127)
    gray_image[(channel1 == 1) & (channel2 == 0)] = 127

    # (1, 1) -> 黑色 (0)
    gray_image[(channel1 == 1) & (channel2 == 1)] = 0

    # 将灰度图转换为 PIL 图像
    gray_image_pil = Image.fromarray(gray_image, mode='L')  # 'L' 模式表示单通道图像

    # 保存图像为 PNG 文件
    gray_image_pil.save(path)
