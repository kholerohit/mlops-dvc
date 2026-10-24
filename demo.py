import json

from ml_pipeline.orchestrator import run_pipeline


def main() -> None:
	result = run_pipeline()
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
