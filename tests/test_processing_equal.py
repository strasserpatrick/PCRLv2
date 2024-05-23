from pathlib import Path

import numpy as np

root_dir = Path(__file__).parent.parent
ground_truth_dir = root_dir / "BraTS_ground_truth"
comparison_dir = root_dir / "BraTS_preprocessed"


def test_processing_equal():
    # TODO: do the execution of the code here as well

    len_gt = len(list(ground_truth_dir.glob("*.npy")))
    len_comp = len(list(comparison_dir.glob("*.npy")))

    assert len_gt == len_comp

    gt_file_names = [f.name for f in ground_truth_dir.glob("*.npy")]
    for f in comparison_dir.glob("*.npy"):
        assert f.name in gt_file_names

        # assert file content equality
        gt_arr = np.load(str(ground_truth_dir / f.name))
        comp_arr = np.load(str(comparison_dir / f.name))

        assert np.array_equal(gt_arr, comp_arr)
