3D version of "Stochastic Segmentation with Conditional Categorical Diffusion Models" at [CCDM](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation)

# Train command
`accelerate launch --num_processes $NGPU --num_machines 1 --main_process_port 6066 main.py -cfg $CFG_FILE`

# Inference command
refer to that in `OvO1111/ldm.git` by constructing an inference config file with
```
model:
    target: inference.models.InferCategoricalDiffusion
    params:
        ...
        use_legacy: true
        parameterization: kl 
        ckpt_path: ...
        ...
```

# Write a config.yaml
Similar to that of the examples under `configs/**.yaml`:
```
dataset:
    train:
        target: ???         # module of the Dataset class
        params:
            kwargs_of_dataset
    val:
        ...

model:
    (tune this to modify the model setting)

encoder:
    context_encoder:
        target: ???         # module of the encoder of crossattn context
        params:
            ...
    condition_encoder:
        ...

trainer:
    (tune this to modify trainer)
```