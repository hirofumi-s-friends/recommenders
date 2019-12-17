import argparse

from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.studio.core.io.data_frame_directory import load_model_from_directory, \
    load_data_frame_from_directory, save_data_frame_to_directory


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        help='The SAR model.',
    )

    parser.add_argument(
        '--users', type=float,
        help='Users to recommend for.',
    )

    parser.add_argument(
        '--item-count-to-recommend', type=str,
        help='Recommend this number of items.',
    )

    parser.add_argument(
        '--output',
        help='The recommended items.',
    )

    known_args, _ = parser.parse_known_args()
    return known_args


if __name__ == '__main__':
    args = get_args()

    model = load_model_from_directory(args.model).model
    input_df = load_data_frame_from_directory(args.users).data

    result = model.recommend_k_items(
        input_df,
        top_k=args.item_count_to_recommend,
    )

    save_data_frame_to_directory(
        args.output,
        result,
        schema=DataFrameSchema.data_frame_to_dict(result),
    )
