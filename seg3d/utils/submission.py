import zlib
import numpy as np

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_submission_pb2, segmentation_metrics_pb2


def compress_array(array: np.ndarray, is_int32: bool = False):
    """Compress a numpy array to ZLIP compressed serialized MatrixFloat/Int32.

    Args:
    array: A numpy array.
    is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

    Returns:
    The compressed bytes.
    """
    if is_int32:
        m = open_dataset.MatrixInt32()
    else:
        m = open_dataset.MatrixFloat()
    m.shape.dims.extend(list(array.shape))
    m.data.extend(array.reshape([-1]).tolist())
    return zlib.compress(m.SerializeToString())


def construct_seg_frame(pred_labels, points_ri, filename):
    top_lidar_row_num = 64
    top_lidar_col_num = 2650

    # assign the dummy class to all valid points (in the range image)
    range_image_pred = np.zeros(
        (top_lidar_row_num, top_lidar_col_num, 2), dtype=np.int32)
    range_image_pred_ri2 = np.zeros(
        (top_lidar_row_num, top_lidar_col_num, 2), dtype=np.int32)

    for i in range(pred_labels.shape[0]):
        if points_ri[i, 2] == 0:
            range_image_pred[points_ri[i, 1], points_ri[i, 0], 1] = pred_labels[i].item() + 1
        elif points_ri[i, 2] == 1:
            range_image_pred_ri2[points_ri[i, 1], points_ri[i, 0], 1] = pred_labels[i].item() + 1

    # construct the segmentationFrame proto.
    context_name_timestamp = filename.split('-')
    context_name = context_name_timestamp[0]
    timestamp_micros = int(context_name_timestamp[1])

    segmentation_frame = segmentation_metrics_pb2.SegmentationFrame()
    segmentation_frame.context_name = context_name
    segmentation_frame.frame_timestamp_micros = timestamp_micros
    laser_semseg = open_dataset.Laser()
    laser_semseg.name = open_dataset.LaserName.TOP
    laser_semseg.ri_return1.segmentation_label_compressed = compress_array(
        range_image_pred, is_int32=True)
    laser_semseg.ri_return2.segmentation_label_compressed = compress_array(
        range_image_pred_ri2, is_int32=True)
    segmentation_frame.segmentation_labels.append(laser_semseg)
    return segmentation_frame


def write_submission_file(segmentation_frame_list, out_file):
    # create the submission file, which can be uploaded to the eval server.
    submission = segmentation_submission_pb2.SemanticSegmentationSubmission()
    submission.account_name = 'wangyang9113@gmail.com'
    submission.unique_method_name = 'WNet'
    submission.affiliation = 'WPCLab'
    submission.authors.append('Darren Wang')
    submission.description = "Proposed by WPCLab"
    submission.method_link = 'NA'
    submission.sensor_type = 1
    submission.number_past_frames_exclude_current = 0
    submission.number_future_frames_exclude_current = 0
    submission.inference_results.CopyFrom(segmentation_frame_list)

    f = open(out_file, 'wb')
    f.write(submission.SerializeToString())
    f.close()
