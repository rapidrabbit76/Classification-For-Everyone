from argparse import ArgumentParser
import pytorch_lightning as pl
def hyperparameters():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    add = parser.add_argument

    ds_candidate = list(DATAMODULE_TABLE.keys())
    model_candidate = list(MODEL_TABLE.keys())
    transfoms_candidate = list(TRANSFORMS_TABLE.keys())

    # experiment hyperparameters
    ## experiment
    add("--seed", type=int, default=9423)
    add("--experiment_name", type=str)
    add("--root_dir", type=str)
    add(
        "--artifact_save_to_logger",
        type=str,
        default="True",
        choices=["True", "False"],
    )

    ## data module/set/transforms
    add("--dataset", type=str, choices=ds_candidate)
    add("--transforms", type=str, choices=transfoms_candidate)
    add("--num_workers", type=int, default=16)
    add("--image_channels", type=int, default=3)
    add("--image_size", type=int)
    add("--batch_size", type=int, default=64)

    ## each model
    add("--model", type=str, choices=model_candidate)
    add("--model_type", type=str)
    add("--num_classes", type=int)
    add("--dropout_rate", type=float, default=0.5)

    ## callbacks
    add("--callbacks_verbose", type=bool, default=True)
    add("--callbacks_monitor", type=str, default="val/acc")
    add("--callbacks_mode", type=str, default="max")
    add("--earlystooping_min_delta", type=float, default=0.1)
    add("--earlystooping_patience", type=float, default=10)

    ## optimizer
    add("--lr", type=float, default=0.1)

    ### SGD
    add("--momentum", type=float, default=0)
    add("--weight_decay", type=float, default=0)
    add("--nesterov", type=float, default=False)

    return parser
