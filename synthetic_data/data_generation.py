import neurokit2 as nk
import pandas as pd

sampling_rate = 4   # Hz (like Empatica E4 from WESAD)
fs = 64     # Hz (like Empatica E4 BVP from WESAD)
duration = 18 * 60   # 3 minutes

# synthetic EDA
eda_signal = nk.eda_simulate(duration=duration, sampling_rate=sampling_rate)

# synthetic BVP 
bvp_signal = nk.ppg_simulate(duration=duration, sampling_rate=fs, heart_rate=70)

df_eda = pd.DataFrame({"EDA": eda_signal})
df_eda.to_csv("synthetic_eda.csv", index=False)

df_bvp = pd.DataFrame({"BVP": bvp_signal})
df_bvp.to_csv("synthetic_bvp.csv", index=False)

print("Synthetic EDA saved to synthetic_eda.csv")

print("Synthetic BVP saved to synthetic_bvp.csv")

