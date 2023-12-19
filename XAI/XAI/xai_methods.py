import numpy as np
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries


def normalize_img(img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.min() < 0 or img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min())
    elif img.dtype == np.uint8:
        if img.min() < 0 or img.max() > 255:
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
    return img


def lime(img, model):
    test_image_np = np.array(img)
    test_image_normalized = normalize_img(test_image_np)

    segmenter = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(test_image_normalized, model, top_labels=1, hide_color=0, num_samples=100, segmentation_fn=segmenter)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

    overlayed_image = mark_boundaries(temp, mask)

    return overlayed_image