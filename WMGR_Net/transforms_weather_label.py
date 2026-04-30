import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np


class WeatherDegradation(ImageOnlyTransform):
    """模拟恶劣天气效果（雨雾/沙尘/暴风雪）"""

    def __init__(self,
                 intensity: float = 0.5,
                 weather_type: str = 'fog',
                 always_apply=False,
                 p=0.5):

        super(WeatherDegradation, self).__init__(always_apply, p)
        self.intensity = np.clip(intensity, 0, 1)
        self.weather_type = weather_type

    def apply(self, image, **params):
        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]

        if self.weather_type == 'fog':
            # 模拟雾效
            fog = np.full_like(image, 200 * self.intensity)
            return cv2.addWeighted(image, 1 - self.intensity, fog, self.intensity, 0)

        elif self.weather_type == 'rain':
            # 模拟雨滴
            overlay = image.copy()
            num_drops = int(500 * self.intensity)
            for _ in range(num_drops):
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                length = np.random.randint(5, 15)
                angle = np.random.uniform(0, 180)
                cv2.line(overlay, (x, y),
                         (x + int(length * np.sin(angle)),
                          y + int(length * np.cos(angle))),
                         (200, 200, 200), 1, cv2.LINE_AA)
            return cv2.addWeighted(image, 0.8, overlay, 0.2, 0)

        elif self.weather_type == 'snow':
            # 模拟雪花
            snow = np.zeros_like(image)
            num_flakes = int(300 * self.intensity)
            for _ in range(num_flakes):
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                size = np.random.randint(2, 5)
                cv2.circle(snow, (x, y), size, (255, 255, 255), -1)
            return cv2.addWeighted(image, 0.9, snow, 0.4, 0)

        return image


    def get_transform_init_args_names(self):
        return ("intensity", "weather_type")

# transforms_weather.py
# 保留 WeatherDegradation 类不变

# 新增：定义单独的变换对象，不要用 Compose 包裹
def get_weather_augmentations():
    return {
        1: WeatherDegradation(intensity=0.6, weather_type='fog', p=1.0),
        2: WeatherDegradation(intensity=0.8, weather_type='rain', p=1.0),
        3: WeatherDegradation(intensity=0.7, weather_type='snow', p=1.0)
    }

class Cut(ImageOnlyTransform):
    def __init__(self,
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting

    def apply(self, image, **params):
        if self.cutting:
            image = image[self.cutting:-self.cutting, :, :]

        return image

    def get_transform_init_args_names(self):
        return ("size", "cutting")


def get_transforms_train(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0):

    # 添加恶劣天气增强选项
    weather_aug = A.OneOf([
        WeatherDegradation(intensity=0.6, weather_type='fog', p=0.7),
        WeatherDegradation(intensity=0.8, weather_type='rain', p=0.5),
        WeatherDegradation(intensity=0.7, weather_type='snow', p=0.4),
        A.RandomToneCurve(scale=0.3, p=0.5)  # 增加色调变化模拟光照变化
    ], p=0.5)  # 50%的概率应用恶劣天气变换

    satellite_transforms = A.Compose([
        A.ImageCompression(quality_lower=80, quality_upper=95, p=0.5),  # 降低质量范围模拟传输损失
        weather_aug,
        # ... 其他变换保持不变 ...
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2 * image_size_sat[0]),
                            max_width=int(0.2 * image_size_sat[0]),
                            min_holes=10,
                            min_height=int(0.1 * image_size_sat[0]),
                            min_width=int(0.1 * image_size_sat[0]),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   weather_aug,  # 地面视图同样应用天气变换
                                   # ... 其他变换保持不变 ...
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1],
                                            interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                 always_apply=False, p=0.5),
                                   A.OneOf([
                                       A.AdvancedBlur(p=1.0),
                                       A.Sharpen(p=1.0),
                                   ], p=0.3),
                                   A.OneOf([
                                       A.GridDropout(ratio=0.5, p=1.0),
                                       A.CoarseDropout(max_holes=25,
                                                       max_height=int(0.2 * img_size_ground[0]),
                                                       max_width=int(0.2 * img_size_ground[0]),
                                                       min_holes=10,
                                                       min_height=int(0.1 * img_size_ground[0]),
                                                       min_width=int(0.1 * img_size_ground[0]),
                                                       p=1.0),
                                   ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    return satellite_transforms, ground_transforms

# 验证集变换保持不变（不添加天气增强）
def get_transforms_val(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0):
    satellite_transforms = A.Compose(
        [A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
         A.Normalize(mean, std),
         ToTensorV2(),
         ])

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1],
                                            interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    return satellite_transforms, ground_transforms