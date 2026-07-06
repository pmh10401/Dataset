import os
import sys

sys.path.append("/Users/minho/Documents/Dataset")

def main():
    print("--- Cross-Domain VAE Dry Run ---")
    try:
        import run_all_adaptive_cnn_cross_domain_contrastive as cd
        res = cd.run_evaluation("ACSF1_normal_2", epochs=1)
        if res is not None:
            print("Success! Cross-Domain VAE runs perfectly.")
            for k, v in res.items():
                if k != "dataset_name":
                    print(f"  * {k:20s}: {v}")
        else:
            print("Failed: Result is None")
    except Exception as e:
        print(f"Error during dry run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
