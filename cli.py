import argparse
from astro_pipe.core.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Astro Pipeline CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)

if __name__ == "__main__":
    main()
