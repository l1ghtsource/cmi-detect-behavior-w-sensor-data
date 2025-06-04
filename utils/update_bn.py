import torch

@torch.no_grad()
def ts_update_bn(loader, model, device):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None

    for batch_input_dict in loader:
        if not isinstance(batch_input_dict, dict):
            raise TypeError('uhhh thats sad!')
            
        imu_data = batch_input_dict.get('imu')
        thm_data = batch_input_dict.get('thm')
        tof_data = batch_input_dict.get('tof')

        if not all(isinstance(data, torch.Tensor) for data in [imu_data, thm_data, tof_data]):
            print('uhhh thats sad!')
            continue

        if device is not None:
            imu_data = imu_data.to(device)
            thm_data = thm_data.to(device)
            tof_data = tof_data.to(device)

        model(imu_data, thm_data, tof_data)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    
    model.train(was_training)