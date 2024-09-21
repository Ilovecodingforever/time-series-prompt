
# https://discuss.pytorch.org/t/finding-source-of-nan-in-forward-pass/51153/3
def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        if out is None:
            continue
        if not isinstance(out, torch.Tensor):

            for j, o in enumerate(out):
                if torch.isnan(out[o]).any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", torch.isnan(out[o]).nonzero(), "where:", o[torch.isnan(out[o]).nonzero()[:, 0].unique(sorted=True)])

                if torch.isinf(out[o]).any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found INF in output {i} at indices: ", torch.isinf(out[o]).nonzero(), "where:", o[torch.isinf(out[o]).nonzero()[:, 0].unique(sorted=True)])
        else:
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

            inf_mask = torch.isinf(out)
            if inf_mask.any():
                print("In", self.__class__.__name__)
                raise RuntimeError(f"Found INF in output {i} at indices: ", inf_mask.nonzero(), "where:", out[inf_mask.nonzero()[:, 0].unique(sorted=True)])
