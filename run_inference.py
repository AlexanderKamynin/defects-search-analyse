import yaml
import argparse
import subprocess
from pathlib import Path
import os
import datetime


class Inference:
    def __init__(self) -> None:
        pass

    @staticmethod
    def run_inference_scripts(scripts: list, log_file: str) -> None:
        for script in scripts:
            start_time = None
            end_time = None
            try:
                print(f"\nStarting inference for instrument: {script['name']}")
                print(f"Script: {script['script']}")
                print(f"Arguments: {' '.join(script['args'])}")
                script_path = Path(script["script"]).resolve()
                script_dir = script_path.parent
                print(f"Working directory: {script_dir}")

                cmd = ["python", str(Path(script["script"]).resolve())] + [
                    str(arg) for arg in script["args"]
                ]

                start_time = datetime.datetime.now()
                result = subprocess.run(cmd, check=True, cwd=script_dir)
                end_time = datetime.datetime.now()
            except Exception as e:
                end_time = datetime.datetime.now()
                print(f"Unexpected error running script: {script['script']} - {str(e)}")
                continue

            if start_time and end_time:
                duration = end_time - start_time
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        line = f"{start_time.isoformat()},{end_time.isoformat()},{script['name']},{duration.total_seconds():.2f}\n"
                        f.write(line)
                    print(
                        f"Inference for {script['name']} take {duration.total_seconds():.2f} seconds."
                    )
                except Exception as e:
                    print(f"Failed to log time for {script['name']}: {str(e)}")

        print(
            "Inference done for all scripts. Check results in output directory, that you point in the YAML configuration file."
        )


class InstrumentsConfig:
    def __init__(self, config: dict = None) -> None:
        self.config = config

    def load_config(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config = yaml.safe_load(config_file)

        self.process_paths()

        print(f"Config file succefully parsed from file: {config_path}")
        self.print_config_details()

    def validate_config(self) -> bool:
        requirements_fields = ["default_output_dir", "instruments"]
        for field in requirements_fields:
            if field not in self.config:
                raise ValueError(f"Requirement field '{field}' not in config")
        print(f"Config contains all requirement fields: passed")
        return True

    def get_config(self) -> dict:
        return self.config

    def print_config_details(self) -> None:
        if not self.config:
            print("Config is empty")
            return

        print("\n=== Global Configuration ===")
        print(
            f"Default output directory: {self.config.get('default_output_dir', 'Not specified')}"
        )
        print(f"Number of instruments: {len(self.config.get('instruments', []))}")

        print("\n=== Instruments Details ===")
        for idx, instrument in enumerate(self.config.get("instruments", []), 1):
            print(f"\nInstrument #{idx}:")
            print(f"  Name: {instrument.get('name', 'Unnamed')}")
            print(
                f"  Description: {instrument.get('description', 'No description specified')}"
            )
            print(f"  Script: {instrument.get('script')}")
            print(
                f"  Result directory: {instrument.get('output_dir', self.config.get('default_output_dir'))}"
            )

            if instrument.get("args"):
                print("  Instrument arguments:")
                for arg, value in instrument.get("args", {}).items():
                    print(f"    {arg}: {value}")

            if instrument.get("metadata"):
                print("  Metadata:")
                for meta, meta_value in instrument.get("metadata", {}).items():
                    print(f"    {meta}: {meta_value}")
        print("===" * 3)

    def prepare_arguments(self, config: dict, images_dir: str) -> list:
        instruments_args = []

        for instrument in config["instruments"]:
            instrument_args = {
                "name": instrument["name"],
                "script": instrument["script"],
                "args": [
                    "--input",
                    images_dir,
                    "--output",
                    instrument.get("output_dir", config["default_output_dir"]),
                ],
            }
            if instrument.get("args"):
                for arg, value in instrument["args"].items():
                    if isinstance(value, bool):
                        if value:
                            instrument_args["args"].append(f"--{arg}")
                    else:
                        instrument_args["args"].extend([f"--{arg}", str(value)])

            instruments_args.append(instrument_args)

        return instruments_args

    def process_paths(self) -> None:
        if "default_output_dir" in self.config:
            self.config["default_output_dir"] = str(
                Path(self.config["default_output_dir"]).resolve()
            )

        for instrument in self.config.get("instruments", []):
            if "script" in instrument:
                instrument["script"] = str(Path(instrument["script"]).resolve())

            if "output_dir" in instrument:
                instrument["output_dir"] = str(Path(instrument["output_dir"]).resolve())
            elif "default_output_dir" in self.config:
                instrument["output_dir"] = self.config["default_output_dir"]


def main():
    parser = argparse.ArgumentParser(description="Run instruments inference")

    parser.add_argument(
        "--config", type=str, required=True, help="path to YAML config file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="path to images directory to inference models",
    )
    args = parser.parse_args()

    instruments_config = InstrumentsConfig()
    instruments_config.load_config(args.config)

    try:
        instruments_config.validate_config()
        config = instruments_config.get_config()

        script_dir = Path(__file__).parent.resolve()
        log_file = script_dir / "inference_times.log"
        if not log_file.exists():
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("start_time,end_time,instrument_name,duration_seconds\n")

        images_path = str(Path(args.images_dir).resolve())
        script_args = instruments_config.prepare_arguments(config, images_path)

        Inference.run_inference_scripts(script_args, log_file=log_file)
    except ValueError as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
