import numpy as np


def calculate_results(mask, pred, num_classes):
    results = []
    true = mask == pred
    for c in range(num_classes):
        result = {}
        mask_c = mask == c
        pred_c = pred == c
        true_mask_c = true[mask_c]
        true_pred_c = true[pred_c]

        area_mask = len(true_mask_c)
        area_pred = len(true_pred_c)
        inter = np.sum(true_mask_c)
        union = area_mask + area_pred - inter

        result["mask"] = area_mask
        result["pred"] = area_pred
        result["inter"] = inter
        result["union"] = union

        results.append(result)

    return results
