# Pytorch implementation of GraphCast. 
This is a pytorch implementation of google's GraphCast with example training script and ready to use dataloader.

## Things I have changed
To fit the model in a device with 40GB of memory, I excluded the destination node feature in the edge_mlp. Checkpointing is applied on every singler layer, please modify base on your system setup. 

## Data setup
Data need to be stored in zarr format. Data need to be placed in the following manner:

    # -- data_root
    #    --atomospherical_data_dir
    #       --1979.zarr
    #       --.....zarr
    #       --2018.zarr
    #    --surface_data_dir
    #       --1979.zarr
    #       --.....zarr
    #       --2018.zarr 

modify the config file in the config directory to modify resolution/pressure_level/variables. 

## This is an unofficial implementation. There might be bugs in the code base, use at your own risks. 