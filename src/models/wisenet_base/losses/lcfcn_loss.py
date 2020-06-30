def lcfcnLoss(model, batch, visualize=False):
    
    model.train()
    N =  batch["images"].size(0)
    assert N == 1

    blob_dict = helpers.get_blob_dict(model, batch, training=True)
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)

    fp_loss = 0.
    split_loss = 0.
    point_loss = 0.
    
    # IMAGE LOSS
    image_loss = helpers.compute_image_loss(S, counts)
    image_loss += helpers.compute_bg_loss(S, counts)

    # POINT LOSS
    point_loss = F.nll_loss(S_log, points, 
                   ignore_index=0, size_average=False)
    import ipdb; ipdb.set_trace()  # breakpoint e5eac4fa //
    
    # FP loss
    if blob_dict["n_fp"] > 0:
        fp_loss = helpers.compute_fp_loss(S_log, blob_dict)

    # Split loss
    if blob_dict["n_multi"] > 0:
        split_loss = helpers.compute_split_loss(S_log, S, points, blob_dict, split_mode="water")

    # Global loss
    split_loss += helpers.compute_boundary_loss(S_log, S, points, counts, add_bg=True)

    return (image_loss + point_loss + fp_loss + split_loss) / N