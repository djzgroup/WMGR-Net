import os
import torch
from dataclasses import dataclass

from torch.utils.data import DataLoader
from WMGR_Net.dataset.cvact_weather import CVACTDatasetTest, CVACTDatasetEval
from WMGR_Net.transforms import get_transforms_val
from WMGR_Net.evaluate.cvusa_and_cvact import evaluate
from WMGR_Net.model_wmgr import WMGR_Net


@dataclass
class Configuration:
    # Model
    model: str = 'vit_base_r50_s16_224'
    # model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 224

    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True

    # Dataset
    data_folder_val = "/smalldata/CVACT"
    data_folder_test = "/data/CVACT-allday"

    # Checkpoint to start from (update this with your trained model path)
    checkpoint_start = '/home/Sample4Geo-main/Sample4Geo-main/cvactpki_label/vit_base_r50_s16_224/065829/weights_e64_89.0928.pth'

    # set num_workers to 0 if on Windows
    num_workers: int = 1

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# -----------------------------------------------------------------------------#
# Config                                                                      #
# -----------------------------------------------------------------------------#

config = Configuration()

if __name__ == '__main__':

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))

    model = WMGR_Net(config.model,
                      pretrained=False,
                      img_size=config.img_size)

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size

    image_size_sat = (img_size, img_size)

    new_width = config.img_size * 2
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)
    img_size_ground = image_size_sat

    # load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)
    else:
        print("Warning: No checkpoint loaded! Evaluating with random weights.")

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Eval transforms
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )

    # Reference Satellite Images - VAL
    reference_dataset_val = CVACTDatasetEval(data_folder=config.data_folder_val,
                                             split="val",
                                             img_type="reference",
                                             transforms=sat_transforms_val)

    reference_dataloader_val = DataLoader(reference_dataset_val,
                                          batch_size=config.batch_size,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True)

    # Query Ground Images - VAL
    query_dataset_val = CVACTDatasetEval(data_folder=config.data_folder_val,
                                         split="val",
                                         img_type="query",
                                         transforms=ground_transforms_val)

    query_dataloader_val = DataLoader(query_dataset_val,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      shuffle=False,
                                      pin_memory=True)

    print("Reference Images Val:", len(reference_dataset_val))
    print("Query Images Val:", len(query_dataset_val))

    # Reference Satellite Images - TEST
    reference_dataset_test = CVACTDatasetTest(data_folder=config.data_folder_test,
                                              img_type="reference",
                                              transforms=sat_transforms_val)

    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

    # Query Ground Images - TEST
    query_dataset_test = CVACTDatasetTest(data_folder=config.data_folder_test,
                                          img_type="query",
                                          transforms=ground_transforms_val)

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))

    # -----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    # -----------------------------------------------------------------------------#

    print("\n{}[{}]{}".format(30 * "-", "Val CVACT", 30 * "-"))

    r1_val = evaluate(config=config,
                      model=model,
                      reference_dataloader=reference_dataloader_val,
                      query_dataloader=query_dataloader_val,
                      ranks=[1, 5, 10],
                      step_size=1000,
                      cleanup=True)

    print("\n{}[{}]{}".format(30 * "-", "Test CVACT", 30 * "-"))

    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=reference_dataloader_test,
                       query_dataloader=query_dataloader_test,
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
