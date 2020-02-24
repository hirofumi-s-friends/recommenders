import argparse
from distutils.util import strtobool
from enum import Enum
import sys

from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.studio.core.logger import module_logger as logger
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.model_directory import load_model_from_directory, pickle_loader


class ScoreType(Enum):
    ITEM_RECOMMENDATION = 'Item recommendation'
    RATING_PREDICTION = 'Rating prediction'


class RankingMetric(Enum):
    RATING = 'Rating'
    SIMILARITY = 'Similarity'
    POPULARITY = 'Popularity'


class ItemSet(Enum):
    TRAIN_ONLY = 'Items in training set'
    SCORE_ONLY = 'Items in score set'


class ScoreSARModule:
    def __init__(self, trained_model, param_dict):
        logger.info(f"Param keys: {param_dict.keys()}")
        self.model = trained_model
        self.score_type = param_dict.get('Score type')
        self.ranking_metric = param_dict.get("Ranking metric")
        self.top_k = param_dict.get("Top k")
        self.sort_top_k = param_dict.get("Sort top k")
        self.remove_seen_items = param_dict.get("Remove seen items")
        self.normalize = param_dict.get("Normalize")
        self.items_to_predict = param_dict.get("Items to predict")

        # Convert type
        self.top_k = int(self.top_k) if self.top_k else None
        self.sort_top_k = strtobool(self.sort_top_k) if self.sort_top_k else None
        self.remove_seen_items = strtobool(self.remove_seen_items) if self.remove_seen_items else None
        self.normalize = strtobool(self.normalize) if self.normalize else None

    def run(self, dataset_to_score, param_dict=None):
        logger.info(f"Runtime dict: {param_dict}")

        # Dynamic params
        if param_dict:
            if param_dict.get("Top k"):
                self.top_k = param_dict.get("Top k")

        score_type = ScoreType(self.score_type)
        if score_type == ScoreType.ITEM_RECOMMENDATION:
            scored_result = self.recommend_items(input_data=dataset_to_score,
                                                 ranking_metric=RankingMetric(self.ranking_metric),
                                                 top_k=self.top_k,
                                                 sort_top_k=self.sort_top_k,
                                                 remove_seen=self.remove_seen_items,
                                                 normalize=self.normalize)
        elif score_type == ScoreType.RATING_PREDICTION:
            scored_result = self.predict_ratings(input_data=dataset_to_score,
                                                 items_to_predict=ItemSet(self.items_to_predict),
                                                 normalize=self.normalize)
        else:
            raise ValueError(f"Got unexpected score type: {score_type}.")

        return scored_result

    def recommend_items(self, input_data, ranking_metric, top_k, sort_top_k, remove_seen, normalize):
        if ranking_metric == RankingMetric.RATING:
            return self.model.recommend_k_items(test=input_data, top_k=top_k, sort_top_k=sort_top_k,
                                                remove_seen=remove_seen, normalize=normalize)
        if ranking_metric == RankingMetric.SIMILARITY:
            return self.model.get_item_based_topk(items=input_data, top_k=top_k, sort_top_k=sort_top_k)
        if ranking_metric == RankingMetric.POPULARITY:
            return self.model.get_popularity_based_topk(top_k=top_k, sort_top_k=sort_top_k)
        raise ValueError(f"Got unexpected ranking metric: {ranking_metric}.")

    def predict_ratings(self, input_data, items_to_predict, normalize):
        if items_to_predict == ItemSet.TRAIN_ONLY:
            return self.model.predict_training_items(test=input_data, normalize=normalize)
        if items_to_predict == ItemSet.SCORE_ONLY:
            return self.model.predict(test=input_data, normalize=normalize)
        raise ValueError(f"Got unexpected 'items to predict': {items_to_predict}.")


def entrance(*args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--trained-model', help='The directory contains trained SAR model.')
    parser.add_argument(
        '--dataset-to-score', help='Dataset to score')
    parser.add_argument(
        '--score-type', type=str, help='The type of score which the recommender should output')
    parser.add_argument(
        '--items-to-predict', type=str, help='The set of items to predict for test users')
    parser.add_argument(
        '--normalize', type=str, help='Normalize predictions to scale of original ratings')
    parser.add_argument(
        '--ranking-metric', type=str, help='The metric of ranking used in item recommendation')
    parser.add_argument(
        '--top-k', type=int, help='The number of top items to recommend.')
    parser.add_argument(
        '--sort-top-k', type=str, help='Sort top k results.')
    parser.add_argument(
        '--remove-seen-items', type=str, help='Remove items seen in training from recommendation')
    parser.add_argument(
        '--score-result', help='Ratings or items to output')

    args, _ = parser.parse_known_args(args)
    logger.info(f"Arguments: {args}")

    trained_model = load_model_from_directory(args.trained_model, model_loader=pickle_loader).data
    param_dict = {
        'Score type': args.score_type,
        "Ranking metric": args.ranking_metric,
        "Sort top k": args.sort_top_k,
        "Top k": args.top_k,
        "Remove seen items": args.remove_seen_items,
        "Normalize": args.normalize,
        "Items to predict": args.items_to_predict,
    }

    score_sar_module = ScoreSARModule(trained_model, param_dict)

    dataset_to_score = load_data_frame_from_directory(args.dataset_to_score).data
    logger.debug(f"Shape of loaded DataFrame: {dataset_to_score.shape}")

    scored_result = score_sar_module.run(dataset_to_score, param_dict)

    save_data_frame_to_directory(args.score_result, data=scored_result,
                                 schema=DataFrameSchema.data_frame_to_dict(scored_result))


if __name__ == '__main__':
    entrance(*sys.argv[1:])
