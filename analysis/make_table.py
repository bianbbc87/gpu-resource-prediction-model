import json
import glob
import numpy as np

def collect(exp_dir):
    mape = []
    rmspe = []

    for f in glob.glob(f"{exp_dir}/seed_*/metrics.json"):
        with open(f) as fp:
            d = json.load(fp)
        mape.append(d["final_mape"])
        rmspe.append(d["final_rmspe"])

    return {
        "MAPE_mean": np.mean(mape),
        "MAPE_std": np.std(mape),
        "RMSPE_mean": np.mean(rmspe),
        "RMSPE_std": np.std(rmspe),
    }

if __name__ == "__main__":
    result = collect("outputs/perfseer_a100")
    print(result)
