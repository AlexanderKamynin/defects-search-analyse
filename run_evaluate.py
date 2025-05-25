import yaml
import argparse
import os
import json
import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path


class DataLoader:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.ground_truth_path = Path(config["ground_truth"]["base_path"]).resolve()
        self.images_extensions = [".png", ".jpg", ".bmp", ".jpeg"]

    def load_real_masks(self) -> pd.DataFrame:
        real_masks = []

        for class_info in self.config["ground_truth"]["classes"]:
            class_name = class_info["name"]
            mask_dir = os.path.join(self.ground_truth_path, class_info["dir"], "masks")

            if not os.path.exists(mask_dir):
                print(f"Directory {mask_dir} did not found.")
                continue

            for mask_file in tqdm(sorted(os.listdir(mask_dir)), desc=f"Loading {class_name} masks from path: {mask_dir}"):
                if os.path.splitext(mask_file)[-1].lower() in self.images_extensions:
                    mask_path = os.path.join(mask_dir, mask_file)
                    mask_name = os.path.splitext(os.path.basename(mask_path))[0]
                    real_masks.append(
                        {
                            "name": mask_name,
                            "mask_path": mask_path,
                            "class": class_name,
                        }
                    )

        return pd.DataFrame(real_masks)

    def load_instruments_masks(self) -> dict:
        predicted_masks_df = {}

        for instrument in self.config["instruments"]:
            name = instrument["name"]
            predictions_path = instrument["predictions_path"]

            df = self.load_instrument_masks(predictions_path)
            predicted_masks_df[name] = df
        return predicted_masks_df

    def load_instrument_masks(self, prediction_path: str) -> pd.DataFrame:
        predicted_masks = []

        if not os.path.exists(prediction_path):
            print(f"Prediction {prediction_path} did not found.")
            return None

        for predicted_mask in tqdm(sorted(os.listdir(prediction_path)), desc=f"Loading predicted masks from path={prediction_path}"):
            if os.path.splitext(predicted_mask)[-1].lower() in self.images_extensions:
                mask_path = os.path.join(prediction_path, predicted_mask)
                mask_name = os.path.splitext(os.path.basename(mask_path))[0]
                predicted_masks.append(
                    {
                        "name": mask_name,
                        "mask_path": mask_path,
                    }
                )

        return pd.DataFrame(predicted_masks)

    @staticmethod
    def load_binary_mask(mask_path: str) -> np.ndarray:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Mask not found at {mask_path}.")
        _, mask = cv2.threshold(mask, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

        return (mask > 0).astype(np.uint8)


class Metrics:
    def __init__(self) -> None:
        pass

    @staticmethod
    def calc_confusion_matrix(
        real_masks_df: pd.DataFrame,
        predicted_masks_dict: dict,
        invert_classes: dict,
    ) -> dict:
        # TP, TN, FP, FN

        for instrument_name, pred_df in predicted_masks_dict.items():
            pred_df["TP"] = 0
            pred_df["FP"] = 0
            pred_df["TN"] = 0
            pred_df["FN"] = 0
            pred_df["class"] = "no class"

            for idx, row in pred_df.iterrows():
                name = row["name"]
                pred_mask_path = row["mask_path"]

                real_mask_row = real_masks_df[real_masks_df["name"] == name]
                if real_mask_row.empty:
                    raise ValueError(f"Real mask for '{name}' not found.")

                real_class = real_mask_row.iloc[0]["class"]
                real_mask_path = real_mask_row.iloc[0]["mask_path"]

                pred_mask = DataLoader.load_binary_mask(pred_mask_path)
                real_mask = DataLoader.load_binary_mask(real_mask_path)

                tp = np.sum((real_mask == 1) & (pred_mask == 1))
                fp = np.sum((real_mask == 0) & (pred_mask == 1))
                tn = np.sum((real_mask == 0) & (pred_mask == 0))
                fn = np.sum((real_mask == 1) & (pred_mask == 0))

                if invert_classes.get(real_class) is True:
                    tp, fp, tn, fn = tn, fn, tp, fp

                pred_df.at[idx, "TP"] = tp
                pred_df.at[idx, "FP"] = fp
                pred_df.at[idx, "TN"] = tn
                pred_df.at[idx, "FN"] = fn
                pred_df.at[idx, "class"] = real_class

            predicted_masks_dict[instrument_name] = pred_df

        return predicted_masks_dict

    @staticmethod
    def calc_accuracy(predicted_masks_dict: dict) -> dict:
        # Pixel Accuracy = TP + TN / (TP + TN + FP + FN)
        for instrument_name, pred_df in predicted_masks_dict.items():
            pred_df["Accuracy"] = (pred_df["TP"] + pred_df["TN"]) / (
                pred_df["TP"] + pred_df["TN"] + pred_df["FP"] + pred_df["FN"]
            ).replace(np.nan, 0)
        return predicted_masks_dict

    @staticmethod
    def calc_iou(predicted_masks_dict: dict) -> dict:
        # IoU (Jaccard Index) = TP / (TP + FP + FN)
        for instrument_name, pred_df in predicted_masks_dict.items():
            union = pred_df["TP"] + pred_df["FP"] + pred_df["FN"]
            intersection = pred_df["TP"]
            pred_df["IoU"] = (intersection / union).replace(np.nan, 0)
        return predicted_masks_dict

    @staticmethod
    def calc_f1_score(predicted_masks_dict: dict) -> dict:
        # F1-score = 2TP / (2TP + FP + FN)
        for instrument_name, pred_df in predicted_masks_dict.items():
            pred_df["F1"] = (
                (2 * pred_df["TP"])
                / (2 * pred_df["TP"] + pred_df["FP"] + pred_df["FN"])
            ).replace(np.nan, 0)
        return predicted_masks_dict

    @staticmethod
    def calc_precision(predicted_masks_dict: dict) -> dict:
        # Precision = TP / (TP + FP)
        for instrument_name, pred_df in predicted_masks_dict.items():
            pred_df["Precision"] = (
                pred_df["TP"] / (pred_df["TP"] + pred_df["FP"])
            ).replace(np.nan, 0)
        return predicted_masks_dict

    @staticmethod
    def calc_recall(predicted_masks_dict: dict) -> dict:
        # Recall = TP / (TP + FN)
        for instrument_name, pred_df in predicted_masks_dict.items():
            pred_df["Recall"] = (
                pred_df["TP"] / (pred_df["TP"] + pred_df["FN"])
            ).replace(np.nan, 0)
        return predicted_masks_dict

    @staticmethod
    def call_all_metrics(predicted_masks_dict: dict) -> dict:
        Metrics.calc_accuracy(predicted_masks_dict)
        Metrics.calc_precision(predicted_masks_dict)
        Metrics.calc_recall(predicted_masks_dict)
        Metrics.calc_iou(predicted_masks_dict)
        Metrics.calc_f1_score(predicted_masks_dict)

        return predicted_masks_dict

    @staticmethod
    def calc_macro_averaging(predicted_masks_df: dict) -> pd.DataFrame:
        # macro_data = []
        # for instrument_name, pred_df in predicted_masks_df.items():
        #     macro_row = {
        #         'Instrument': instrument_name,
        #         'Accuracy': pred_df['Accuracy'].mean(),
        #         'Precision': pred_df['Precision'].mean(),
        #         'Recall': pred_df['Recall'].mean(),
        #         'F1': pred_df['F1'].mean(),
        #         'IoU': pred_df['IoU'].mean()
        #     }
        #     macro_data.append(macro_row)

        macro_data = []

        combined = pd.concat(
            [df.assign(instrument=name) for name, df in predicted_masks_df.items()],
            ignore_index=True,
        )

        grouped = (
            combined.groupby(["instrument", "class"])
            .agg(
                {
                    "Accuracy": "mean",
                    "Precision": "mean",
                    "Recall": "mean",
                    "F1": "mean",
                    "IoU": "mean",
                }
            )
            .reset_index()
        )

        grouped.columns = [
            "Instrument",
            "Class",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "IoU",
        ]

        # return pd.DataFrame(macro_data).set_index('Instrument')
        return grouped.set_index(["Instrument", "Class"])

    @staticmethod
    def calc_micro_averaging(predicted_masks_df: dict) -> pd.DataFrame:
        # micro_data = []

        # for instrument_name, pred_df in predicted_masks_df.items():
        #     total_tp = pred_df["TP"].sum()
        #     total_fp = pred_df['FP'].sum()
        #     total_tn = pred_df['TN'].sum()
        #     total_fn = pred_df['FN'].sum()

        #     accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        #     precision = total_tp / (total_tp + total_fp)
        #     recall = total_tp / (total_tp + total_fn)
        #     f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
        #     iou = total_tp / (total_tp + total_fp + total_fn)

        #     micro_row = {
        #         'Instrument': instrument_name,
        #         'Accuracy': accuracy,
        #         'Precision': precision,
        #         'Recall': recall,
        #         'F1': f1,
        #         'IoU': iou
        #     }
        #     micro_data.append(micro_row)

        # return pd.DataFrame(micro_data).set_index("Instrument")
        micro_data = []

        for instrument_name, df in predicted_masks_df.items():
            grouped = (
                df.groupby("class")
                .agg({"TP": "sum", "FP": "sum", "TN": "sum", "FN": "sum"})
                .reset_index()
            )

            eps = 1e-10
            for _, row in grouped.iterrows():
                tp = row["TP"]
                fp = row["FP"]
                tn = row["TN"]
                fn = row["FN"]

                accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1 = 2 * tp / (2 * tp + fp + fn + eps)
                iou = tp / (tp + fp + fn + eps)

                micro_data.append(
                    {
                        "Instrument": instrument_name,
                        "Class": row["class"],
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1,
                        "IoU": iou,
                    }
                )

        return pd.DataFrame(micro_data).set_index(["Instrument", "Class"])


class ResultsVisualizer:
    images_extensions = [".png", ".jpg", ".bmp", ".jpeg"]
    
    def __init__(self):
        pass

    @staticmethod
    def generate(
        config: dict,
        real_masks_df: pd.DataFrame,
        predicted_masks_dict: dict,
        macro_metrics: pd.DataFrame,
        micro_metrics: pd.DataFrame,
    ):
        output_dir = Path(config["default_output_dir"])
        formats = config.get("metrics_format", ["csv"])

        ResultsVisualizer.save_formatted(
            data=macro_metrics,
            base_name="macro_metrics",
            output_dir=output_dir,
            formats=formats,
        )
        ResultsVisualizer.save_formatted(
            data=micro_metrics,
            base_name="micro_metrics",
            output_dir=output_dir,
            formats=formats,
        )

        for instrument_name, df in predicted_masks_dict.items():
            safe_name = ResultsVisualizer._sanitize_filename(instrument_name)
            ResultsVisualizer.save_formatted(
                data=df,
                base_name=f"{safe_name}_detailed",
                output_dir=output_dir / "detailed",
                formats=config["metrics_format"],
            )

        if config.get("save_side_by_side", False):
            for instrument_name, df in predicted_masks_dict.items():
                safe_name = ResultsVisualizer._sanitize_filename(instrument_name)
                ResultsVisualizer.save_side_by_side(
                    config=config,
                    instrument_name=instrument_name,
                    pred_df=df,
                    real_masks_df=real_masks_df,
                    output_dir=output_dir,
                )

    @staticmethod
    def save_formatted(
        data: pd.DataFrame, base_name: str, output_dir: str, formats: list
    ):
        os.makedirs(output_dir, exist_ok=True)
        for fmt in formats:
            full_path = output_dir / f"{base_name}.{fmt}"
            if fmt == "csv":
                data.to_csv(full_path, index=True)
            elif fmt == "json":
                data.to_json(full_path, orient="split", indent=4)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        return (
            name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "")
            .lower()
        )

    @staticmethod
    def save_side_by_side(
        config: dict,
        instrument_name: str,
        pred_df: pd.DataFrame,
        real_masks_df: pd.DataFrame,
        output_dir: str,
    ):
        images_dir = output_dir / "side_by_side" / instrument_name
        images_dir.mkdir(parents=True, exist_ok=True)
        base_path = Path(config["ground_truth"]["base_path"])

        for _, pred_row in pred_df.iterrows():
            mask_name = pred_row["name"]
            true_mask_row = real_masks_df[real_masks_df["name"] == mask_name]
            if true_mask_row.empty:
                print(f"Warning: True mask for '{mask_name}' not found. Skipping.")
                continue
            true_mask_path = true_mask_row.iloc[0]["mask_path"]
            class_name = true_mask_row.iloc[0]["class"]

            true_mask = DataLoader.load_binary_mask(true_mask_path) * 255
            pred_mask = DataLoader.load_binary_mask(pred_row["mask_path"]) * 255
            # print(f"path: {true_mask_path}, val={np.unique(DataLoader.load_binary_mask(pred_row['mask_path']))}")
            image = None
            for class_info in config["ground_truth"]["classes"]:
                if class_info["name"] == class_name:
                    image_dir = base_path / class_info["dir"] / "images"
                    for ext in ResultsVisualizer.images_extensions:
                        image_path = image_dir / f"{mask_name}{ext}"
                        if image_path.exists():
                            image = cv2.imread(str(image_path))
                            break
                    break
                

            if image is not None:
                h, w = image.shape[:2]
                side_by_side = np.zeros((h, w * 3, 3), dtype=np.uint8)
                side_by_side[:, :w] = image
                side_by_side[:, w : 2 * w] = cv2.cvtColor(true_mask, cv2.COLOR_GRAY2BGR)
                side_by_side[:, 2 * w :] = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
            else:
                h, w = true_mask.shape
                side_by_side = np.zeros((h, w * 2, 3), dtype=np.uint8)
                side_by_side[:, :w] = cv2.cvtColor(true_mask, cv2.COLOR_GRAY2BGR)
                side_by_side[:, w:] = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

            output_path = images_dir / f"{mask_name}.jpg"
            cv2.imwrite(str(output_path), side_by_side)


class ConfigParser:
    def __init__(self, config_path: str, output_dir: str = None) -> None:
        self.config_path = config_path
        self.output_dir = output_dir
        self.config = None

    def load_config(self) -> dict:
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        if self.output_dir:
            self.config["default_output_dir"] = self.output_dir

        self.process_paths()
        self.validate()

        return self.config

    def validate(self) -> bool:
        requirements_fields = [
            "ground_truth",
            "instruments",
            "default_output_dir",
            "metrics_format",
            "save_side_by_side"
        ]
        for field in requirements_fields:
            if field not in self.config:
                raise ValueError(f"Requirement field '{field}' not in config")
        print(f"Config contains all requirement fields: passed")

        for instrument in self.config["instruments"]:
            if "predictions_path" not in instrument:
                instrument_name = instrument["name"]
                print(
                    f"Warnings: Instrument '{instrument_name}' doesn't have 'predictions_path' field. Metrics will be not calculated for this instrument."
                )
        return True

    def process_paths(self) -> None:
        if "default_output_dir" in self.config:
            output_dir = Path(self.config["default_output_dir"]).resolve()
            self.config["default_output_dir"] = str(output_dir)
            output_dir.mkdir(exist_ok=True)
            print(f"Output directory created: {output_dir}")

        for instrument in self.config.get("instruments", []):
            if "predictions_path" in instrument:
                instrument["predictions_path"] = str(
                    Path(instrument["predictions_path"]).resolve()
                )


def main():
    parser = argparse.ArgumentParser(description="Run evalute the instruments results")

    parser.add_argument(
        "--config",
        type=str,
        default="./evaluate_config.yaml",
        help="path to YAML config file",
    )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="./eval_results",
    #     help="output path for evaluation result",
    # )
    args = parser.parse_args()

    # ----------------------------------------
    config_handler = ConfigParser(args.config)
    config = config_handler.load_config()
    # ----------------------------------------

    # ----------------------------------------
    dataloader = DataLoader(config)
    real_masks_df = dataloader.load_real_masks()
    predicted_masks_dict = dataloader.load_instruments_masks()
    # ----------------------------------------

    print("Calculate metrics...")
    # ----------------------------------------
    invert_classes = {  # Словарь классов с инверсией
        cls["name"]: cls.get("invert", False)
        for cls in config["ground_truth"]["classes"]
    }
    predicted_masks_dict = Metrics.calc_confusion_matrix(
        real_masks_df=real_masks_df,
        predicted_masks_dict=predicted_masks_dict,
        invert_classes=invert_classes,
    )
    predicted_masks_dict = Metrics.call_all_metrics(
        predicted_masks_dict=predicted_masks_dict
    )
    macro_metrics = Metrics.calc_macro_averaging(predicted_masks_dict)
    micro_metrics = Metrics.calc_micro_averaging(predicted_masks_dict)
    # ----------------------------------------
    

    print("Generate report...")
    ResultsVisualizer.generate(
        config=config,
        macro_metrics=macro_metrics,
        micro_metrics=micro_metrics,
        real_masks_df=real_masks_df,
        predicted_masks_dict=predicted_masks_dict,
    )
    
    print(f"Evaluate succefully! Check out the output directory: {config['default_output_dir']}")


if __name__ == "__main__":
    main()
