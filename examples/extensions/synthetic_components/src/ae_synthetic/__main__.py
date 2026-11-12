"""Create a scene or exercise the installed stateless extension."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("scene", "events"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "scene":
        from .scenes import provider_scene
        value = provider_scene()
    else:
        from arch_explorer.adf import evaluate
        from .scenes import event_objects
        value = evaluate(*event_objects(), backend="events").to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
