from dataset.pose_dataset import PoseDataset


def create(cfg):
    dataset_type = cfg.dataset_type
    if dataset_type == "mpii":
        from dataset.mpii import MPII
        data = MPII(cfg)
    elif dataset_type == "coco":
        print("jin ru coco")
        from dataset.mscoco import MSCOCO
        print("jie su coco")
        data = MSCOCO(cfg)
    elif dataset_type == "penn_action":
        from dataset.penn_action import PennAction
        data = PennAction(cfg)
    elif dataset_type == "default":
        data = PoseDataset(cfg)
    else:
        raise Exception("Unsupported dataset_type: \"{}\"".format(dataset_type))
    return data
